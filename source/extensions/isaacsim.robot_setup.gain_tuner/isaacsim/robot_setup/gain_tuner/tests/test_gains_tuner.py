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

"""Unit tests for the GainTuner core (inertia, test runners, results API)."""

from __future__ import annotations

import math
from collections.abc import Generator

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni.kit.test
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.gains_tuner import (
    _assign_d6_axis_token,
    _d6_axis_has_unlocked_limit,
    _extract_d6_axis_token,
    _format_d6_display_name,
    classify_joint_modes,
    compute_parallel_axis_inertia,
    find_articulation_root,
    get_joint_axis_world_direction,
    matrix_norm,
    resolve_active_source_gains,
)
from omni.physics.tensors import DofType
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from .common import (
    DriveSubmodality,
    JointModality,
    TestGainTunerHarness,
    _compute_natural_freq_damping_prismatic,
    _compute_natural_freq_damping_revolute,
    _revolute_drive_stiffness_damping_usd_to_si,
)


class TestGainTuner(TestGainTunerHarness):
    """Equivalent inertia from articulation tensors vs. hand-computed I_eq."""

    # ---- Unit tests for compute_joints_accumulated_inertia ----
    # Hand-computed expected equivalent inertia and optional stiffness/damping from natural frequency
    # are asserted against the implementation. Relative tolerance 5% covers articulation mass-property
    # tensors vs. the rigid-body hand model (any backend), not the closed-form helpers tested elsewhere.

    async def test_compute_joints_accumulated_inertia_fixed_base_single_revolute(self) -> None:
        """Fixed base + single revolute: I_eq = link inertia about joint axis (backward fixed)."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        # Joint is at base (0,0,0.5); link COM at (0.5,0,0.5). I_eq = I_cm + m*d^2 = 1 + 1*0.5^2 = 1.25
        expected_I_eq = 1.0 + 1.0 * (0.5**2)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 1)
        computed = self._gain_tuner._joint_accumulated_inertia.get(entries[0].joint)
        self.assertIsNotNone(computed, "Should have computed inertia for single revolute")
        self.assertAlmostEqual(computed, expected_I_eq, delta=0.05 * max(expected_I_eq, 1e-6))
        # Stiffness/damping from natural frequency: K = I*w_n^2, D = 2*zeta*I*w_n
        stiffness_attr = UsdPhysics.DriveAPI(entries[0].joint, "angular").GetStiffnessAttr()
        damping_attr = UsdPhysics.DriveAPI(entries[0].joint, "angular").GetDampingAttr()
        K, D = stiffness_attr.Get(), damping_attr.Get()
        K, D = _revolute_drive_stiffness_damping_usd_to_si(K, D)
        nat_freq, zeta = _compute_natural_freq_damping_revolute(K, D, computed)
        # Gains were authored with inertia_diag=1.0 but I_eq=1.25; recovered nat_freq = 10*sqrt(1/1.25)
        expected_nat_freq = 10.0 * math.sqrt(1.0 / expected_I_eq)
        self.assertAlmostEqual(nat_freq, expected_nat_freq, delta=0.2)
        self.assertAlmostEqual(zeta, 0.05, delta=0.02)

    async def test_compute_joints_accumulated_inertia_fixed_base_single_prismatic(self) -> None:
        """Fixed base + single prismatic: I_eq = link mass (backward fixed)."""
        robot_path = self._create_articulation(
            [JointModality.PRISMATIC],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=2.0,
            inertia_diag=1.0,
            natural_freq_hz=5.0,
            damping_ratio=0.1,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        expected_I_eq = 2.0  # mass
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 1)
        computed = self._gain_tuner._joint_accumulated_inertia.get(entries[0].joint)
        self.assertIsNotNone(computed)
        self.assertAlmostEqual(computed, expected_I_eq, delta=0.05 * expected_I_eq)
        stiffness_attr = UsdPhysics.DriveAPI(entries[0].joint, "linear").GetStiffnessAttr()
        damping_attr = UsdPhysics.DriveAPI(entries[0].joint, "linear").GetDampingAttr()
        K, D = stiffness_attr.Get(), damping_attr.Get()
        nat_freq, zeta = _compute_natural_freq_damping_prismatic(K, D, computed)
        self.assertAlmostEqual(nat_freq, 5.0, delta=0.3)
        self.assertAlmostEqual(zeta, 0.1, delta=0.03)

    async def test_compute_joints_accumulated_inertia_fixed_base_two_revolute_same_plane(self) -> None:
        """Fixed base + two revolute joints in the same plane: hand-computed I_eq for each joint."""
        distance = 0.5
        mass, inertia_diag = 1.0, 1.0
        robot_path, expected_list = self._create_fixed_base_two_revolute_chain(
            distance=distance,
            mass=mass,
            inertia_diag=inertia_diag,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
            second_axis_z=True,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 2)
        for entry, expected in zip(entries, expected_list):
            computed = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
            self.assertIsNotNone(computed, f"Inertia for {entry.joint.GetPath()}")
            self.assertAlmostEqual(computed, expected, delta=0.05 * max(expected, 1e-6))

    async def test_compute_joints_accumulated_inertia_fixed_base_two_revolute_orthogonal(self) -> None:
        """Fixed base + two revolute joints in orthogonal planes."""
        distance = 0.5
        mass, inertia_diag = 1.0, 1.0
        robot_path, expected_list = self._create_fixed_base_two_revolute_chain(
            distance=distance,
            mass=mass,
            inertia_diag=inertia_diag,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
            second_axis_z=False,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 2)
        for entry, expected in zip(entries, expected_list):
            computed = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
            self.assertIsNotNone(computed)
            self.assertAlmostEqual(computed, expected, delta=0.05 * max(expected, 1e-6))

    async def test_compute_joints_accumulated_inertia_moving_base_single_revolute(self) -> None:
        """Moving base + single revolute: I_eq = (I_base * I_link_about_joint) / (I_base + I_link_about_joint).

        Implementation uses inertia about the joint: link contributes I_d + m*d^2.
        """
        distance = 0.5
        base_inertia, link_inertia_diag, link_mass = 1.0, 1.0, 1.0
        robot_path, _ = self._create_moving_base_single_joint(
            joint_revolute=True,
            distance=distance,
            base_mass=10.0,
            base_inertia=base_inertia,
            link_mass=link_mass,
            link_inertia_diag=link_inertia_diag,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        # Link inertia about joint (joint on base): I_d + m*d^2
        I_link_about_joint = link_inertia_diag + link_mass * (distance**2)
        expected_I_eq = (base_inertia * I_link_about_joint) / (base_inertia + I_link_about_joint)  # 5/9
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 1)
        computed = self._gain_tuner._joint_accumulated_inertia.get(entries[0].joint)
        self.assertIsNotNone(computed)
        self.assertAlmostEqual(computed, expected_I_eq, delta=0.05 * max(expected_I_eq, 1e-6))

    async def test_compute_joints_accumulated_inertia_fixed_base_two_prismatic_same_axis(self) -> None:
        """Fixed base + two prismatic joints on the same axis: j0 I_eq = m0+m1, j1 I_eq = m0*m1/(m0+m1)."""
        robot_path, expected_list = self._create_fixed_base_two_prismatic_chain(
            distance=0.5,
            mass=1.0,
            natural_freq_hz=5.0,
            damping_ratio=0.1,
            same_axis=True,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 2)
        for entry, expected in zip(entries, expected_list):
            computed = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
            self.assertIsNotNone(computed)
            self.assertAlmostEqual(computed, expected, delta=0.05 * max(expected, 1e-6))

    async def test_compute_joints_accumulated_inertia_fixed_base_two_prismatic_orthogonal(self) -> None:
        """Fixed base + two prismatic joints on orthogonal axes."""
        robot_path, expected_list = self._create_fixed_base_two_prismatic_chain(
            distance=0.5,
            mass=1.0,
            natural_freq_hz=5.0,
            damping_ratio=0.1,
            same_axis=False,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 2)
        for entry, expected in zip(entries, expected_list):
            computed = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
            self.assertIsNotNone(computed)
            self.assertAlmostEqual(computed, expected, delta=0.05 * max(expected, 1e-6))

    async def test_compute_joints_accumulated_inertia_moving_base_single_prismatic(self) -> None:
        """Moving base + single prismatic: I_eq = (m_base * m_link) / (m_base + m_link)."""
        robot_path, expected_list = self._create_moving_base_single_joint(
            joint_revolute=False,
            distance=0.5,
            base_mass=10.0,
            base_inertia=1.0,
            link_mass=1.0,
            link_inertia_diag=1.0,
            natural_freq_hz=5.0,
            damping_ratio=0.1,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        expected_I_eq = (10.0 * 1.0) / (10.0 + 1.0)  # 10/11
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 1)
        computed = self._gain_tuner._joint_accumulated_inertia.get(entries[0].joint)
        self.assertIsNotNone(computed)
        self.assertAlmostEqual(computed, expected_I_eq, delta=0.05 * max(expected_I_eq, 1e-6))

    async def test_compute_joints_accumulated_inertia_fixed_base_revolute_prismatic_chain(self) -> None:
        """Fixed base + revolute then prismatic.

        ``j0`` I_eq = inertia of (link0+link1) about axis; ``j1`` I_eq = m0*m1/(m0+m1).
        """
        robot_path, expected_list = self._create_fixed_base_revolute_prismatic_chain(
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 2)
        for entry, expected in zip(entries, expected_list):
            computed = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
            self.assertIsNotNone(computed)
            self.assertAlmostEqual(computed, expected, delta=0.05 * max(expected, 1e-6))


class _RecordTrajectoryTest(gain_tuner.RobotTest):
    """Records articulation state for N physics steps.

    Args:
        num_steps: Number of physics steps to record.
    """

    name = "RecordTrajectory"

    def __init__(self, num_steps: int) -> None:
        super().__init__()
        self._num_steps = num_steps

    def setup(
        self,
        articulation: object,
        joint_indices: list[int],
        joint_modes: dict[int, object],
        test_params: dict[str, object],
    ) -> None:
        super().setup(articulation, joint_indices, joint_modes, test_params)
        self._joint_indices = joint_indices

    def run(self) -> Generator[None, None, gain_tuner.TestResult]:
        cmd_p: list[np.ndarray] = []
        cmd_v: list[np.ndarray] = []
        obs_p: list[np.ndarray] = []
        obs_v: list[np.ndarray] = []
        times: list[float] = []
        t = 0.0
        for _ in range(self._num_steps):
            cmd_p.append(self._articulation.get_dof_position_targets().numpy()[0].copy())
            cmd_v.append(self._articulation.get_dof_velocity_targets().numpy()[0].copy())
            obs_p.append(self._articulation.get_dof_positions().numpy()[0].copy())
            obs_v.append(self._articulation.get_dof_velocities().numpy()[0].copy())
            times.append(t)
            t += self.step
            yield
        times_arr = np.array(times)
        return gain_tuner.TestResult(
            joint_position_commands=np.stack(cmd_p),
            joint_velocity_commands=np.stack(cmd_v),
            observed_joint_positions=np.stack(obs_p),
            observed_joint_velocities=np.stack(obs_v),
            command_times=times_arr,
            joint_metrics={0: {"steps": self._num_steps}},
        )


CUSTOM_GAINS_TEST_MODE = 99


class TestGainTunerRobotTestPath(TestGainTunerHarness):
    """Registered :class:`RobotTest` returning :class:`TestResult`."""

    async def test_registered_robot_test_populates_buffers(self) -> None:
        """Registered robot tests populate command and observation buffers."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        self._gain_tuner.register_test(CUSTOM_GAINS_TEST_MODE, _RecordTrajectoryTest(5))
        try:
            await self._run_setup_and_compute_inertia(robot_path)
            self._gain_tuner.initialize_gains_test(
                {
                    "test_mode": CUSTOM_GAINS_TEST_MODE,
                    "joint_indices": [0],
                    "test_duration": 0.1,
                    "sequence": [],
                    "steps": 5,
                }
            )
            self._timeline.play()
            dt = self._physics_dt
            for _ in range(30):
                done = self._gain_tuner.update_gains_test(dt)
                await app_utils.update_app_async()
                if done:
                    break
            self._timeline.stop()
            pos_cmd, _, _, _, times = self._gain_tuner.get_joint_states_from_gains_test(0)
            self.assertIsNotNone(pos_cmd)
            self.assertGreater(pos_cmd.size, 0)
            self.assertIsNotNone(times)
            metrics = self._gain_tuner.get_test_result_metrics()
            self.assertEqual(metrics[0].get("steps"), 5)
        finally:
            self._gain_tuner.unregister_test(CUSTOM_GAINS_TEST_MODE)


class TestActuatorMjcTestClassification(omni.kit.test.AsyncTestCase):
    """Part C: actuator- / mjc-active joints are classified POSITION and commanded."""

    _ROOT = "/World/Robot"

    def _make_multi_source_stage(self):
        """In-memory articulation: 3 revolute joints (cubes) + a Newton actuator + an mjc joint.

        Mirrors ``_create_articulation`` (cubes + revolute joints + ArticulationRootAPI)
        without a live sim: joint 0 has DriveAPI gains, joint 1 a Newton PD actuator,
        joint 2 MuJoCo-native gains authored on the joint.
        """
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        root = stage.DefinePrim(self._ROOT, "Xform")
        UsdPhysics.ArticulationRootAPI.Apply(root)
        joints = {}
        for i in range(3):
            link = UsdGeom.Cube.Define(stage, f"{self._ROOT}/link_{i}")
            UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
            UsdPhysics.RevoluteJoint.Define(stage, f"{self._ROOT}/joint_{i}")
            joints[i] = stage.GetPrimAtPath(f"{self._ROOT}/joint_{i}")
        # joint_0: DriveAPI position gains.
        d0 = UsdPhysics.DriveAPI.Apply(joints[0], "angular")
        d0.CreateStiffnessAttr(100.0)
        d0.CreateDampingAttr(10.0)
        # joint_1: Newton PD actuator (its DriveAPI gains are zeroed by the runtime).
        stage.DefinePrim(f"{self._ROOT}/Actuators", "Scope")
        act = stage.DefinePrim(f"{self._ROOT}/Actuators/act1", "Xform")
        act.AddAppliedSchema("NewtonPDControlAPI")
        act.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(300.0)
        act.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(30.0)
        act.CreateRelationship("newton:targets").AddTarget(Sdf.Path(f"{self._ROOT}/joint_1"))
        # joint_2: MuJoCo-native position gains authored directly on the joint.
        joints[2].CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set([1500.0] + [0.0] * 9)
        joints[2].CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set([0.0, -1500.0, -50.0] + [0.0] * 7)
        return stage, joints

    def _maps(self, stage):
        return gain_tuner.build_actuator_gain_map(stage, self._ROOT), gain_tuner.build_mjc_gain_map(stage)

    async def test_resolve_active_source_gains_actuator_and_mjc(self) -> None:
        """Actuator (always) and mjc (under the MuJoCo solver) yield effective gains."""
        stage, joints = self._make_multi_source_stage()
        actuator_map, mjc_map = self._maps(stage)
        dof_joint_map = {i: joints[i] for i in range(3)}
        overrides = resolve_active_source_gains(
            [0, 1, 2], dof_joint_map, actuator_map, mjc_map, mujoco_solver_active=True
        )
        self.assertNotIn(0, overrides)  # DriveAPI joint keeps its tensor gains
        self.assertEqual(overrides[1], (300.0, 30.0))
        self.assertAlmostEqual(overrides[2][0], 1500.0)
        self.assertAlmostEqual(overrides[2][1], 50.0)

    async def test_resolve_active_source_gains_mjc_needs_mujoco_solver(self) -> None:
        """The mjc joint is only active (overridden) when the MuJoCo solver runs."""
        stage, joints = self._make_multi_source_stage()
        actuator_map, mjc_map = self._maps(stage)
        dof_joint_map = {i: joints[i] for i in range(3)}
        overrides = resolve_active_source_gains(
            [0, 1, 2], dof_joint_map, actuator_map, mjc_map, mujoco_solver_active=False
        )
        self.assertNotIn(2, overrides)  # mjc not active without the MuJoCo solver
        self.assertIn(1, overrides)  # actuator is backend-independent

    async def test_classify_marks_actuator_and_mjc_position(self) -> None:
        """Actuator- and mjc-active joints classify POSITION even with zero DriveAPI gains."""
        stage, joints = self._make_multi_source_stage()
        actuator_map, mjc_map = self._maps(stage)
        dof_joint_map = {i: joints[i] for i in range(3)}
        overrides = resolve_active_source_gains(
            [0, 1, 2], dof_joint_map, actuator_map, mjc_map, mujoco_solver_active=True
        )
        # The physics tensor reports DriveAPI gains; the actuator / mjc joints read
        # 0 there because their DriveAPI gains are zeroed by the actuator runtime.
        modes = classify_joint_modes([0, 1, 2], [100.0, 0.0, 0.0], [10.0, 0.0, 0.0], overrides)
        self.assertEqual(modes[0], gain_tuner.JointMode.POSITION)
        self.assertEqual(modes[1], gain_tuner.JointMode.POSITION)
        self.assertEqual(modes[2], gain_tuner.JointMode.POSITION)

    async def test_actuator_mjc_joints_dropped_without_override(self) -> None:
        """Without the override the actuator / mjc joints would be dropped as un-driven."""
        modes = classify_joint_modes([0, 1, 2], [100.0, 0.0, 0.0], [10.0, 0.0, 0.0], {})
        self.assertEqual(modes[0], gain_tuner.JointMode.POSITION)
        self.assertEqual(modes[1], gain_tuner.JointMode.NONE)
        self.assertEqual(modes[2], gain_tuner.JointMode.NONE)

    async def test_velocity_classification_from_damping_only(self) -> None:
        """A DOF whose active source has only a damping-like gain classifies VELOCITY."""
        modes = classify_joint_modes([0], [0.0], [0.0], {0: (0.0, 25.0)})
        self.assertEqual(modes[0], gain_tuner.JointMode.VELOCITY)

    async def test_newton_mujoco_solver_active_defensive(self) -> None:
        """MuJoCo-solver detection returns a bool and never raises (defensive)."""
        self.assertIsInstance(gain_tuner.newton_mujoco_solver_active(), bool)

    async def test_all_test_types_command_actuator_mjc_joints(self) -> None:
        """Built-in, snap, and stress all select POSITION-mode joints for commanding."""
        modes = {0: gain_tuner.JointMode.POSITION, 1: gain_tuner.JointMode.POSITION, 2: gain_tuner.JointMode.VELOCITY}
        seq_joint_indices = [0, 1, 2]
        # Built-in sinusoidal / step path (GainTuner._partition_joints_by_mode).
        tuner = gain_tuner.GainTuner()
        tuner.joint_modes = modes
        tuner.test_params = {"sequence": [{"joint_indices": seq_joint_indices}]}
        pos_dof_idx, vel_dof_idx, _, _ = tuner._partition_joints_by_mode(0)
        self.assertEqual(pos_dof_idx, [0, 1])
        self.assertEqual(vel_dof_idx, [2])
        # snap_to_limits.py and stress_test.py use the identical mode-based
        # selection (``joint_modes.get(i, NONE) == POSITION`` for pos DOFs), so
        # POSITION-classified actuator / mjc joints are commanded by them too.
        snap_stress_pos = [
            int(i)
            for i in seq_joint_indices
            if modes.get(int(i), gain_tuner.JointMode.NONE) == gain_tuner.JointMode.POSITION
        ]
        self.assertEqual(snap_stress_pos, [0, 1])


class TestGainTunerBuiltInCommands(TestGainTunerHarness):
    """Built-in sinusoidal / step command generators on GainTuner."""

    def _sinusoidal_test_params(self, dof_index: int = 0) -> dict:
        return {
            "test_mode": gain_tuner.GainsTestMode.SINUSOIDAL,
            "joint_indices": [dof_index],
            "test_duration": 1.0,
            "sequence": [
                {
                    "joint_indices": np.array([dof_index], dtype=np.int32),
                    "joint_amplitudes": np.array([0.5], dtype=np.float32),
                    "joint_offsets": np.array([0.0], dtype=np.float32),
                    "joint_periods": np.array([2.0], dtype=np.float32),
                    "joint_phases": np.array([0.0], dtype=np.float32),
                    "joint_step_max": np.array([0.5], dtype=np.float32),
                    "joint_step_min": np.array([-0.5], dtype=np.float32),
                    "joint_user_provided": [False],
                }
            ],
        }

    async def test_sinusoidal_step_returns_position_commands(self) -> None:
        """Sinusoidal command generation returns position commands."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
            joint_limit_revolute=(-90.0, 90.0),
        )
        await self._run_setup_and_compute_inertia(robot_path)
        self._gain_tuner.initialize_gains_test(self._sinusoidal_test_params(0))
        pos_idx, pos_cmd, vel_idx, vel_cmd = self._gain_tuner.sinusoidal_step(0.25, 0)
        self.assertEqual(len(pos_idx), 1)
        self.assertEqual(len(pos_cmd), 1)
        self.assertIsInstance(vel_idx, list)

    async def test_step_step_square_wave_changes_with_timestep(self) -> None:
        """Step command generation flips square-wave position commands over time."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
            joint_limit_revolute=(-90.0, 90.0),
        )
        await self._run_setup_and_compute_inertia(robot_path)
        params = self._sinusoidal_test_params(0)
        params["test_mode"] = gain_tuner.GainsTestMode.STEP
        self._gain_tuner.initialize_gains_test(params)
        _, pos_a, _, _ = self._gain_tuner.step_step(0.0, 0)
        _, pos_b, _, _ = self._gain_tuner.step_step(1.5, 0)
        self.assertEqual(len(pos_a), 1)
        self.assertAlmostEqual(float(pos_a[0]), 0.5, places=5)
        self.assertAlmostEqual(float(pos_b[0]), -0.5, places=5)

    async def test_builtin_test_registry_contains_sinusoidal_and_step(self) -> None:
        """The built-in test registry exposes sinusoidal and step tests."""
        tuner = gain_tuner.GainTuner()
        tuner.register_test(gain_tuner.GainsTestMode.SINUSOIDAL, gain_tuner.SinusoidalTest())
        tuner.register_test(gain_tuner.GainsTestMode.STEP, gain_tuner.StepFunctionTest())
        names = {t.name for t in tuner.get_registered_tests().values()}
        self.assertIn("Sinusoidal", names)
        self.assertIn("Step Function", names)


class TestGainTunerInternals(TestGainTunerHarness):
    """D6 helpers, articulation root, inertia math, and callbacks."""

    async def test_extract_d6_axis_token_from_dof_name(self) -> None:
        """D6 axis tokens are extracted from DOF names."""
        self.assertEqual(_extract_d6_axis_token("arm:rotZ"), "rotZ")
        self.assertEqual(_extract_d6_axis_token("slide_transX"), "transX")

    async def test_assign_d6_axis_token_avoids_duplicates(self) -> None:
        """Assigned D6 axis tokens avoid duplicates for the same joint."""
        usage: dict[str, set] = {}
        first = _assign_d6_axis_token("joint", "dof_rotX", usage)
        second = _assign_d6_axis_token("joint", "dof_rotX", usage)
        self.assertNotEqual(first, second)

    async def test_d6_axis_has_unlocked_limit(self) -> None:
        """D6 axes with authored finite limits are treated as unlocked."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.Joint.Define(stage, "/d6")
        limit_api = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), "rotZ")
        limit_api.CreateLowAttr(-1.0)
        limit_api.CreateHighAttr(1.0)
        self.assertTrue(_d6_axis_has_unlocked_limit(joint.GetPrim(), "rotZ"))

    async def test_format_d6_display_name(self) -> None:
        """D6 display names include the joint name and axis token."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.Joint.Define(stage, "/arm")
        name = _format_d6_display_name(joint.GetPrim(), "rotZ", "rotZ")
        self.assertEqual(name, "arm:rotZ")

    async def test_find_articulation_root_on_robot_child(self) -> None:
        """Articulation root discovery finds the child root joint."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        root = find_articulation_root(self._stage, robot_path)
        self.assertEqual(root, f"{robot_path}/root_joint")

    async def test_matrix_norm_and_parallel_axis_inertia(self) -> None:
        """Matrix norm and parallel-axis inertia helpers return expected values."""
        m = Gf.Matrix3f(3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0)
        self.assertAlmostEqual(matrix_norm(m), math.sqrt(50.0), places=6)
        I_cm = Gf.Matrix3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        translated = compute_parallel_axis_inertia(I_cm, 2.0, Gf.Vec3f(0.0, 1.0, 0.0))
        self.assertAlmostEqual(translated[0][0], 3.0, places=5)

    async def test_get_joint_axis_world_direction_unit_length(self) -> None:
        """World joint-axis directions are normalized."""
        stage = Usd.Stage.CreateInMemory()
        jpath = "/joint"
        UsdPhysics.RevoluteJoint.Define(stage, jpath).CreateAxisAttr("Z")
        joint = stage.GetPrimAtPath(jpath)
        axis = get_joint_axis_world_direction(joint, Gf.Matrix4d(1.0))
        self.assertAlmostEqual(axis.GetLength(), 1.0, places=5)

    async def test_add_inertia_updated_callback_invoked(self) -> None:
        """Registered inertia-updated callbacks are invoked."""
        tuner = gain_tuner.GainTuner()
        called = []

        def _cb() -> None:
            called.append(1)

        tuner.add_inertia_updated_callback(_cb)
        tuner._notify_inertia_updated()
        self.assertEqual(len(called), 1)


class TestGainTunerResultsIngestAPI(omni.kit.test.AsyncTestCase):
    """Public results-ingest API used by the dt-sweep orchestrator (no private pokes)."""

    async def test_robot_prim_path_defaults_to_none(self) -> None:
        """A freshly constructed tuner reports no robot prim path."""
        tuner = gain_tuner.GainTuner()
        self.assertIsNone(tuner.get_robot_prim_path())

    async def test_ingest_sweep_results_with_trajectory_round_trips(self) -> None:
        """Ingesting metrics + a trajectory exposes them via the normal public getters."""
        tuner = gain_tuner.GainTuner()
        metrics = {0: {"test_type": "discretization_sweep", "status": "accurate"}}
        trajectory = tuple(np.array([[float(i), float(i) + 1.0]]) for i in range(5))
        tuner.ingest_sweep_results(metrics, trajectory)
        self.assertEqual(tuner.get_test_result_metrics(), metrics)
        self.assertTrue(tuner.is_data_ready())
        snapshot = tuner.snapshot_recorded_trajectory()
        self.assertEqual(len(snapshot), 5)
        for restored, original in zip(snapshot, trajectory):
            np.testing.assert_array_equal(restored, original)

    async def test_ingest_sweep_results_metrics_only_leaves_trajectory(self) -> None:
        """A metrics-only ingest replaces the metrics but does not touch the recorded arrays."""
        tuner = gain_tuner.GainTuner()
        trajectory = tuple(np.array([[float(i)]]) for i in range(5))
        tuner.ingest_sweep_results({0: {"status": "accurate"}}, trajectory)
        tuner.ingest_sweep_results({1: {"status": "degraded"}})
        self.assertEqual(tuner.get_test_result_metrics(), {1: {"status": "degraded"}})
        # The trajectory is untouched and data remains ready.
        self.assertTrue(tuner.is_data_ready())
        np.testing.assert_array_equal(tuner.snapshot_recorded_trajectory()[0], trajectory[0])

    async def test_clear_test_results_resets_metrics_and_data_ready(self) -> None:
        """Clearing results empties the metrics and marks the data as not ready."""
        tuner = gain_tuner.GainTuner()
        tuner.ingest_sweep_results({0: {"status": "accurate"}}, tuple(np.array([[1.0]]) for _ in range(5)))
        tuner.clear_test_results()
        self.assertEqual(tuner.get_test_result_metrics(), {})
        self.assertFalse(tuner.is_data_ready())


class TestGainTunerStopTestWithoutArticulation(omni.kit.test.AsyncTestCase):
    """`stop_test` stays safe in the states where no articulation is bound."""

    async def test_stop_test_before_setup_is_a_no_op(self) -> None:
        """Cancelling a test on a tuner that was never set up does not raise."""
        tuner = gain_tuner.GainTuner()
        self.assertIsNone(tuner._articulation)
        tuner.stop_test()

    async def test_stop_test_after_reset_is_a_no_op(self) -> None:
        """`reset` clears the articulation, so a following cancel must not raise."""
        tuner = gain_tuner.GainTuner()
        tuner.reset()
        self.assertIsNone(tuner._articulation)
        tuner.stop_test()

    async def test_stop_test_without_articulation_still_stops_active_test(self) -> None:
        """The active test is torn down even when there is no articulation to restore."""
        tuner = gain_tuner.GainTuner()
        stopped = []

        class _StubTest(gain_tuner.RobotTest):
            def stop(self) -> None:
                stopped.append(1)

        tuner._active_test = _StubTest()
        tuner.stop_test()
        self.assertEqual(len(stopped), 1)
        self.assertIsNone(tuner._active_test)


class _StubVelocityArray:
    """Minimal stand-in for the warp array `get_dof_max_velocities` returns."""

    def __init__(self, value: float) -> None:
        self._value = value

    def numpy(self) -> np.ndarray:
        return np.asarray([self._value], dtype=np.float32)


class _StubArticulation:
    """Articulation stub reporting one DOF's engine-enforced velocity limit."""

    def __init__(self, max_velocity: float, dof_type, raises: bool = False, armature: float = 0.0) -> None:
        self._max_velocity = max_velocity
        self.dof_types = [dof_type]
        self._raises = raises
        self._armature = armature

    def get_dof_max_velocities(self, dof_indices=None) -> _StubVelocityArray:
        if self._raises:
            raise RuntimeError("physics view is not available")
        return _StubVelocityArray(self._max_velocity)

    def get_dof_armatures(self, dof_indices=None) -> _StubVelocityArray:
        if self._raises:
            raise RuntimeError("physics view is not available")
        return _StubVelocityArray(self._armature)


class TestEffectiveMaxVelocity(omni.kit.test.AsyncTestCase):
    """The engine's velocity limit, which the velocity sweeps actually use.

    `step_sinusoid` / `step_step` scale their commands by
    `get_dof_max_velocities`, so the Advanced panel has to be able to see that
    number in the units USD stores to tell whether what it shows is in force.
    """

    async def test_rotational_limit_is_converted_to_degrees_per_second(self) -> None:
        """The tensor API reports radians; both joint schemas store degrees."""
        tuner = gain_tuner.GainTuner()
        tuner._articulation = _StubArticulation(math.pi, DofType.Rotation)
        self.assertAlmostEqual(tuner.get_dof_effective_max_velocity(0), 180.0, places=3)

    async def test_linear_limit_is_reported_unchanged(self) -> None:
        """Prismatic DOFs share their units with USD, so no conversion applies."""
        tuner = gain_tuner.GainTuner()
        tuner._articulation = _StubArticulation(2.5, DofType.Translation)
        self.assertAlmostEqual(tuner.get_dof_effective_max_velocity(0), 2.5, places=5)

    async def test_no_articulation_degrades_to_none(self) -> None:
        """Before setup there is no engine truth to report, and no crash."""
        tuner = gain_tuner.GainTuner()
        self.assertIsNone(tuner._articulation)
        self.assertIsNone(tuner.get_dof_effective_max_velocity(0))

    async def test_unavailable_physics_view_degrades_to_none(self) -> None:
        """The limit is only queryable while a physics scene is loaded."""
        tuner = gain_tuner.GainTuner()
        tuner._articulation = _StubArticulation(1.0, DofType.Rotation, raises=True)
        self.assertIsNone(tuner.get_dof_effective_max_velocity(0))

    async def test_unlimited_engine_limit_reads_as_infinite_not_none(self) -> None:
        """ "No limit" and "no engine truth" are different answers.

        Collapsing them meant the caller could not tell an unclamped joint from one
        it had not been able to ask about, so an unauthored limit against an
        enforced one was reported as agreement.
        """
        tuner = gain_tuner.GainTuner()
        tuner._articulation = _StubArticulation(math.inf, DofType.Rotation)
        self.assertEqual(tuner.get_dof_effective_max_velocity(0), math.inf)

    async def test_a_practically_unlimited_engine_limit_reads_as_infinite(self) -> None:
        """Newton substitutes 1e6 for an unclamped DOF, not infinity.

        ``newton.utils.import_usd`` maps an authored ``inf`` to None and then to
        ``default_joint_cfg.velocity_limit``, so an unlimited joint comes back as a
        finite 1e6 rad/s.  Converted to degrees that is 5.7e7, which would be
        reported as a disagreement against ``inf`` on every unauthored joint.

        The stub reports the engine's own literal 1e6 rather than
        :data:`~isaacsim.robot_setup.gain_tuner.UNLIMITED_VELOCITY_THRESHOLD`.
        Feeding the threshold back in tested ``>= threshold`` against itself, which
        holds for any threshold: raise the constant to 1e7 and a real 1e6 from
        Newton starts being reported as a 5.7e7 deg/s limit, with the test still
        green.
        """
        tuner = gain_tuner.GainTuner()
        tuner._articulation = _StubArticulation(1.0e6, DofType.Rotation)
        self.assertEqual(tuner.get_dof_effective_max_velocity(0), math.inf)
        # And a limit a user could plausibly author is still reported as a limit,
        # so the threshold is not swallowing real values.
        tuner._articulation = _StubArticulation(100.0, DofType.Rotation)
        self.assertLess(tuner.get_dof_effective_max_velocity(0), 1.0e6)

    async def test_engine_armature_is_reported_unscaled(self) -> None:
        """Both schemas store armature in the same units the tensor API uses."""
        tuner = gain_tuner.GainTuner()
        tuner._articulation = _StubArticulation(1.0, DofType.Rotation, armature=0.1)
        self.assertAlmostEqual(tuner.get_dof_engine_armature(0), 0.1, places=6)

    async def test_engine_armature_degrades_to_none(self) -> None:
        """No articulation, no physics view, and no armature tensor all read as None."""
        self.assertIsNone(gain_tuner.GainTuner().get_dof_engine_armature(0))
        tuner = gain_tuner.GainTuner()
        tuner._articulation = _StubArticulation(1.0, DofType.Rotation, raises=True)
        self.assertIsNone(tuner.get_dof_engine_armature(0))


class TestEffectiveMaxVelocityAgainstBackend(TestGainTunerHarness):
    """The engine limit on a live articulation whose two backends are tuned apart.

    Which authored value the engine enforces is decided by the running backend and
    solver -- PhysX reads ``physxJoint:maxJointVelocity``, Newton resolves through
    ``newton:velocityLimit`` first -- so the assertion is phrased against the
    resolution for the active engine rather than against one schema.  Run under
    either backend, this pins the reported effective value to what the velocity
    sweeps will actually command.
    """

    async def test_engine_enforces_the_resolved_value_for_the_active_backend(self) -> None:
        """The reported effective value is the one the engine really enforces."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        spec = gain_tuner.joint_param_spec("max_joint_velocity")
        joint = self._stage.GetPrimAtPath(f"{robot_path}/joint_0")
        joint.ApplyAPI(gain_tuner.NEWTON_JOINT_API)
        joint.ApplyAPI(gain_tuner.PHYSX_JOINT_API)
        joint.GetAttribute(spec.newton_attr).Set(90.0)
        joint.GetAttribute(spec.physx_attr).Set(180.0)

        self._gain_tuner.setup(robot_path)
        for _ in range(2):
            await app_utils.update_app_async()
        self._timeline.play()
        for _ in range(60):
            await app_utils.update_app_async()

        newton_active = (SimulationManager.get_active_physics_engine() or "").lower() == "newton"
        backend = gain_tuner.BACKEND_NEWTON if newton_active else gain_tuner.BACKEND_PHYSX
        solver = gain_tuner.newton_solver_type(self._stage) if newton_active else ""
        resolution = gain_tuner.resolve_joint_param(joint, spec, backend, solver)
        effective = self._gain_tuner.get_dof_effective_max_velocity(0)
        self._timeline.stop()

        self.assertTrue(resolution.determined, f"the resolver chain must be known (solver={solver!r})")
        self.assertIsNotNone(effective, "the engine limit must be readable while playing")
        self.assertAlmostEqual(effective, resolution.effective_value, delta=0.01)
        self.assertTrue(gain_tuner.max_velocity_agrees(resolution.effective_value, effective))
        # The other backend keeps its own, deliberately different limit.
        self.assertAlmostEqual(resolution.other_value, 180.0 if newton_active else 90.0, places=5)


class TestLiveUsdWriteReachesTheEngine(TestGainTunerHarness):
    """Whether authoring an advanced joint param mid-run changes what is simulated.

    The two backends differ, which is why ``backend_reads_usd_while_playing``
    exists.  PhysX picks a mid-run ``physxJoint:*`` edit up on the next step.
    Newton does not: it builds its ``Model`` once, in ``ModelBuilder.add_usd`` at
    initialization, and ``isaacsim.physics.newton`` registers no USD notice
    handler, so a ``newton:*`` edit sits in the stage unread until the next play.
    Neither behaviour is documented, so it is measured here, and the panel's
    "replay to apply" note is gated on the result.

    Run under either backend, these assert that backend's behaviour, so they also
    catch Newton gaining a live-update path -- at which point the note should go.
    """

    async def _play_one_joint(self) -> Usd.Prim:
        """Build, author 0.05 armature / 90 deg-per-s on both schemas, and play."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        joint = self._stage.GetPrimAtPath(f"{robot_path}/joint_0")
        joint.ApplyAPI(gain_tuner.NEWTON_JOINT_API)
        joint.ApplyAPI(gain_tuner.PHYSX_JOINT_API)
        for spec, value in (
            (gain_tuner.joint_param_spec("armature"), 0.05),
            (gain_tuner.joint_param_spec("max_joint_velocity"), 90.0),
        ):
            joint.GetAttribute(spec.newton_attr).Set(value)
            joint.GetAttribute(spec.physx_attr).Set(value)

        self._gain_tuner.setup(robot_path)
        for _ in range(2):
            await app_utils.update_app_async()
        self._timeline.play()
        for _ in range(60):
            await app_utils.update_app_async()
        self._robot_path = robot_path
        return joint

    def _live_backend(self) -> tuple[str, bool]:
        """Return the active backend label and whether it re-reads USD mid-run.

        Fails closed on the label, which is the property these tests exist to
        establish.  Mapping "not newton" to PhysX made the suite assert PhysX's
        measured mid-run behaviour against whatever engine happened to be running,
        including none at all -- the same assumption
        :func:`~isaacsim.robot_setup.gain_tuner.backend_reads_usd_while_playing`
        refuses to make.  An engine outside the two measured ones asserts nothing
        about the write path, so say so rather than pass.
        """
        engine = (SimulationManager.get_active_physics_engine() or "").lower()
        backend = {"newton": gain_tuner.BACKEND_NEWTON, "physx": gain_tuner.BACKEND_PHYSX}.get(engine, engine)
        self.assertTrue(
            gain_tuner.backend_supported(backend),
            f"the mid-run write path is only measured under PhysX and Newton; engine is {engine!r}",
        )
        return backend, gain_tuner.backend_reads_usd_while_playing(backend)

    async def test_a_mid_run_armature_write_reaches_physx_but_not_newton(self) -> None:
        """Armature is the silent case: it has no engine-truth warning of its own."""
        joint = await self._play_one_joint()
        spec = gain_tuner.joint_param_spec("armature")
        parsed = self._gain_tuner.get_dof_engine_armature(0)

        joint.GetAttribute(spec.newton_attr).Set(0.42)
        joint.GetAttribute(spec.physx_attr).Set(0.42)
        for _ in range(30):
            await app_utils.update_app_async()
        after = self._gain_tuner.get_dof_engine_armature(0)
        self._timeline.stop()

        backend, live = self._live_backend()
        self.assertIsNotNone(parsed, f"the engine must report an armature while playing ({backend})")
        self.assertAlmostEqual(parsed, 0.05, delta=0.005, msg=f"the authored armature must be parsed ({backend})")
        self.assertIsNotNone(after)
        expected = 0.42 if live else 0.05
        self.assertAlmostEqual(
            after,
            expected,
            delta=0.005,
            msg=f"{backend} reported {after} for a mid-run write of 0.42 over a parsed 0.05",
        )

    async def test_a_mid_run_velocity_write_reaches_physx_but_not_newton(self) -> None:
        """The velocity limit, which does at least warn once the timeline plays."""
        joint = await self._play_one_joint()
        spec = gain_tuner.joint_param_spec("max_joint_velocity")
        parsed = self._gain_tuner.get_dof_effective_max_velocity(0)

        joint.GetAttribute(spec.newton_attr).Set(37.0)
        joint.GetAttribute(spec.physx_attr).Set(37.0)
        for _ in range(30):
            await app_utils.update_app_async()
        after = self._gain_tuner.get_dof_effective_max_velocity(0)
        self._timeline.stop()

        backend, live = self._live_backend()
        self.assertIsNotNone(parsed, f"the engine must report a velocity limit while playing ({backend})")
        self.assertAlmostEqual(parsed, 90.0, delta=0.5, msg=f"the authored limit must be parsed ({backend})")
        self.assertIsNotNone(after)
        expected = 37.0 if live else 90.0
        self.assertAlmostEqual(
            after,
            expected,
            delta=0.5,
            msg=f"{backend} reported {after} for a mid-run write of 37.0 over a parsed 90.0",
        )

    async def test_a_replay_picks_up_what_was_authored_during_the_previous_run(self) -> None:
        """Which is why the note says to replay rather than that the edit was lost."""
        joint = await self._play_one_joint()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.42)
        joint.GetAttribute(spec.physx_attr).Set(0.42)
        for _ in range(10):
            await app_utils.update_app_async()

        self._timeline.stop()
        for _ in range(10):
            await app_utils.update_app_async()
        self._gain_tuner.invalidate_physics_views()
        self._timeline.play()
        for _ in range(60):
            await app_utils.update_app_async()
        self._gain_tuner.setup(self._robot_path)
        for _ in range(20):
            await app_utils.update_app_async()
        replayed = self._gain_tuner.get_dof_engine_armature(0)
        self._timeline.stop()

        backend, _ = self._live_backend()
        self.assertIsNotNone(replayed, f"the engine must report an armature after the replay ({backend})")
        self.assertAlmostEqual(
            replayed,
            0.42,
            delta=0.005,
            msg=f"{backend} must parse the value authored during the previous run",
        )


class TestMaxVelocityAgreement(omni.kit.test.AsyncTestCase):
    """Comparing an authored velocity limit against the enforced one."""

    async def test_identical_values_agree(self) -> None:
        """The common case: what USD authors is what the engine enforces."""
        self.assertTrue(gain_tuner.max_velocity_agrees(180.0, 180.0))

    async def test_conversion_noise_is_not_a_disagreement(self) -> None:
        """A radian round-trip through float32 must not be reported as a conflict."""
        self.assertTrue(gain_tuner.max_velocity_agrees(180.0, math.degrees(math.radians(180.0)) * (1 + 1e-6)))

    async def test_different_limits_disagree(self) -> None:
        """A joint swept at half the limit the panel shows is the reported case."""
        self.assertFalse(gain_tuner.max_velocity_agrees(180.0, 90.0))

    async def test_unknowable_values_are_not_a_disagreement(self) -> None:
        """Genuinely unknowable comparisons stay quiet.

        None means the value could not be determined at all -- no engine truth
        before the timeline plays, or a resolver chain that depends on an unknown
        solver.  There is nothing to reconcile against either.
        """
        self.assertTrue(gain_tuner.max_velocity_agrees(None, 90.0))
        self.assertTrue(gain_tuner.max_velocity_agrees(180.0, None))
        self.assertTrue(gain_tuner.max_velocity_agrees(None, None))

    async def test_an_unlimited_usd_limit_against_an_enforced_one_disagrees(self) -> None:
        """The safeguard that used to fail open exactly where it was needed.

        An unauthored limit reads as unlimited, and the caller passes ``inf`` for
        it.  Reporting that as agreement disabled the one check that could catch a
        panel showing an unclamped joint the engine is really clamping.
        """
        self.assertFalse(gain_tuner.max_velocity_agrees(math.inf, 107.0))
        self.assertFalse(gain_tuner.max_velocity_agrees(107.0, math.inf))

    async def test_unlimited_on_both_sides_agrees(self) -> None:
        """No limit authored and no limit enforced is the common, quiet case."""
        self.assertTrue(gain_tuner.max_velocity_agrees(math.inf, math.inf))

    async def test_nan_is_not_a_disagreement(self) -> None:
        """NaN compares unequal to everything, including itself."""
        self.assertTrue(gain_tuner.max_velocity_agrees(math.nan, 90.0))
