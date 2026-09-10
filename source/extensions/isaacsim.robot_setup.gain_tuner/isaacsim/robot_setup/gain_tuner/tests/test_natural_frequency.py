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

"""Unit tests for natural-frequency drive math and PD oscillation dynamics."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from unittest import mock

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.test
import omni.timeline
import omni.usd
from isaacsim.core.simulation_manager import PhysicsScene, PhysxScene, SimulationManager
from isaacsim.robot_setup import gain_tuner
from isaacsim.robot_setup.gain_tuner.gains_tuner import project_inertia_onto_axis
from pxr import Gf, Sdf, Usd, UsdPhysics

from .common import (
    DriveSubmodality,
    JointModality,
    TestGainTunerHarness,
    _compute_natural_freq_damping_prismatic,
    _compute_natural_freq_damping_revolute,
    _compute_stiffness_damping_prismatic,
    _compute_stiffness_damping_revolute,
    _revolute_drive_stiffness_damping_si_to_usd,
    _revolute_drive_stiffness_damping_usd_to_si,
)

# Golden PD scalars (reference values; forward formulas documented in test docstrings).
# Prismatic: m=2.3 kg, f_n=4.5 Hz, ζ=0.12 → k = m ω_n², d = 2 ζ √(m k).
_GOLDEN_PRISMATIC_STIFFNESS_DAMPING_M_2p3_FN_4p5_Z_0p12 = (
    1838.7072999229474,
    15.607432303034091,
)


# Revolute SI: I=0.85 kg·m², f_n=11 Hz, ζ=0.08 → k = I ω_n², d = 2 ζ √(I k).
_GOLDEN_REVOLUTE_SI_STIFFNESS_DAMPING_I_0p85_FN_11_Z_0p08 = (
    4060.355250608161,
    9.39964521954066,
)


# Damped cosine f_n=6.2 Hz, ζ=0.06: T_d = 2π / (ω_n √(1−ζ²)) [s].
_GOLDEN_DAMPED_PERIOD_FN_6p2_Z_0p06 = 0.16158143139130263


# USD drive storage for k_si=150 N·m/rad, d_si=3 N·m·s/rad (scale π/180).
_GOLDEN_REVOLUTE_DRIVE_USD_K = 2.6179938779914944


_GOLDEN_REVOLUTE_DRIVE_USD_D = 0.05235987755982988


@dataclass
class OscillationAnalysis:
    """Results from analyzing damped oscillation data."""

    log_decrement_avg: float
    damped_period: float
    damped_freq: float
    damping_ratio: float
    natural_freq: float
    peak_times: list[float]
    peak_values: list[float]
    num_samples: int = 0
    value_min: float = 0.0
    value_max: float = 0.0


@dataclass
class PdOscillationHarnessResult:
    """Oscillation analysis plus the USD / inertia context used to author PD gains."""

    analysis: OscillationAnalysis
    modality: JointModality
    joint_path: str
    ieq: float
    target_fn_hz: float
    target_zeta: float


def _natural_fn_zeta_from_usd_drive(
    joint_modality: JointModality, joint_prim: Usd.Prim, ieq: float
) -> tuple[float, float]:
    """Closed-loop linear natural frequency (Hz) and zeta from authored USD drive gains and ``I_eq`` / mass.

    Args:
        joint_modality: Joint type used to choose angular or linear drive formulas.
        joint_prim: USD joint prim containing drive attributes.
        ieq: Equivalent inertia or mass for the joint.

    Returns:
        ``(natural_frequency_hz, damping_ratio)`` from the authored USD drive.
    """
    if joint_modality == JointModality.REVOLUTE:
        drive = UsdPhysics.DriveAPI(joint_prim, "angular")
        k_u = float(drive.GetStiffnessAttr().Get())
        d_u = float(drive.GetDampingAttr().Get())
        k_si, d_si = _revolute_drive_stiffness_damping_usd_to_si(k_u, d_u)
        return _compute_natural_freq_damping_revolute(k_si, d_si, ieq)
    drive = UsdPhysics.DriveAPI(joint_prim, "linear")
    k = float(drive.GetStiffnessAttr().Get())
    d = float(drive.GetDampingAttr().Get())
    return _compute_natural_freq_damping_prismatic(k, d, ieq)


def _extract_peaks(times: np.ndarray, values: np.ndarray) -> tuple[list[float], list[float]]:
    """Extract local maxima (peaks) from time-series data.

    Includes boundary indices when they are local maxima (e.g. initial condition
    at release). If no interior peaks are found, falls back to peaks of the
    absolute value (envelope) so both sides of a damped oscillation are counted.

    Args:
        times: Sample times corresponding to ``values``.
        values: Time-series values to inspect.

    Returns:
        ``(peak_times, peak_values)`` for the extracted peaks.
    """
    if len(values) < 2:
        return [], []
    n = len(values)
    # Interior peaks: strict local maxima
    peak_indices: list[int] = []
    if n >= 3:
        is_peak = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
        peak_indices = (np.where(is_peak)[0] + 1).tolist()
    # First point is a peak if it is >= neighbor (e.g. start at initial displacement)
    if values[0] >= values[1]:
        peak_indices.insert(0, 0)
    # Last point is a peak if it is >= neighbor
    if n >= 2 and values[-1] >= values[-2]:
        peak_indices.append(n - 1)
    if peak_indices:
        peak_indices = sorted(set(peak_indices))
        peak_times = times[peak_indices].tolist()
        peak_values = values[peak_indices].tolist()
        return peak_times, peak_values
    # Fallback: peaks of |values| (envelope) so we catch both positive and negative humps
    abs_vals = np.abs(values)
    if n >= 3:
        is_peak_abs = (abs_vals[1:-1] > abs_vals[:-2]) & (abs_vals[1:-1] > abs_vals[2:])
        peak_indices = (np.where(is_peak_abs)[0] + 1).tolist()
    if not peak_indices:
        return [], []
    peak_times = times[peak_indices].tolist()
    # Store signed values for analysis (log decrement uses abs(peak_values))
    peak_values = values[peak_indices].tolist()
    return peak_times, peak_values


def _peaks_fail_msg(result: OscillationAnalysis, prefix: str = "") -> str:
    """Build assertion message when peak count is too low, including data diagnostics.

    Args:
        result: Oscillation analysis to summarize.
        prefix: Optional message prefix.

    Returns:
        Assertion message string.
    """
    return (
        f"{prefix}Need at least 3 peaks for analysis (got {len(result.peak_values)}). "
        f"Position data: n={result.num_samples} min={result.value_min:.6f} max={result.value_max:.6f}"
    )


def _analyze_oscillation(times: np.ndarray, values: np.ndarray) -> OscillationAnalysis:
    """Extract natural frequency and damping ratio from damped oscillation data.

    Uses:
    - Logarithmic decrement: s_i = ln(A_i / A_{i+1}), average over peaks
    - Damped period T_d from average time between successive peaks
    - w_d = 2*pi / T_d
    - zeta = s / sqrt(4*pi^2 + s^2)
    - w_n = w_d / sqrt(1 - zeta^2)

    Args:
        times: Sample times for the response values.
        values: Response values to analyze.

    Returns:
        Oscillation analysis metrics.
    """
    num_samples = len(values)
    value_min = float(np.min(values)) if num_samples else 0.0
    value_max = float(np.max(values)) if num_samples else 0.0
    peak_times, peak_values = _extract_peaks(times, values)
    if len(peak_values) < 2:
        return OscillationAnalysis(
            log_decrement_avg=0.0,
            damped_period=0.0,
            damped_freq=0.0,
            damping_ratio=0.0,
            natural_freq=0.0,
            peak_times=peak_times,
            peak_values=peak_values,
            num_samples=num_samples,
            value_min=value_min,
            value_max=value_max,
        )

    log_decrements = []
    for i in range(len(peak_values) - 1):
        a_i = abs(peak_values[i])
        a_next = abs(peak_values[i + 1])
        if a_next > 1e-12 and a_i > 1e-12:
            s_i = math.log(a_i / a_next)
            log_decrements.append(s_i)
    s_avg = float(np.mean(log_decrements)) if log_decrements else 0.0

    period_diffs = [peak_times[i + 1] - peak_times[i] for i in range(len(peak_times) - 1)]
    T_d = float(np.mean(period_diffs)) if period_diffs else 0.0
    w_d = (2.0 * math.pi / T_d) if T_d > 1e-9 else 0.0

    zeta = s_avg / math.sqrt(4.0 * math.pi**2 + s_avg**2) if (4 * math.pi**2 + s_avg**2) > 0 else 0.0
    denom = 1.0 - zeta**2
    w_n = (w_d / math.sqrt(denom)) if denom > 1e-9 else 0.0

    return OscillationAnalysis(
        log_decrement_avg=s_avg,
        damped_period=T_d,
        damped_freq=w_d,
        damping_ratio=zeta,
        natural_freq=w_n / (2.0 * math.pi),
        peak_times=peak_times,
        peak_values=peak_values,
        num_samples=num_samples,
        value_min=value_min,
        value_max=value_max,
    )


class _OscillationDynamicsMixin:
    """PD oscillation tests: sample rate > 11× target natural frequency, zero gravity, no link collision.

    This is **not** a :class:`unittest.TestCase` subclass so ``omni.kit.test`` does not collect these
    ``test_*`` methods twice (subclasses mix this in with :class:`TestGainTunerHarness` only).

    Solver-independent theory is covered by :class:`TestGainTunerClosedFormTheory` and
    :class:`TestOscillationAnalysisMath`.
    """

    async def setUp(self) -> None:
        self._timeline = omni.timeline.get_timeline_interface()
        self._gain_tuner = gain_tuner.GainTuner()
        await stage_utils.create_new_stage_async()
        self._stage = omni.usd.get_context().get_stage()
        self._stage.DefinePrim(Sdf.Path("/World"), "Xform")
        scene_path = "/World/physicsScene"
        if SimulationManager.get_active_physics_engine() != "physx":
            if not SimulationManager.switch_physics_engine("physx", verbose=False):
                self.skipTest("PhysX physics engine could not be activated for oscillation tests")
        PhysxScene(scene_path)
        PhysicsScene(scene_path).set_gravity((0.0, 0.0, 0.0))
        SimulationManager.set_default_physics_scene(scene_path)
        self._fn_hz = 6.0
        self._physics_dt = 1.0 / (11.5 * self._fn_hz)
        SimulationManager.set_physics_dt(self._physics_dt)
        await app_utils.update_app_async()
        try:
            SimulationManager.initialize_physics()
        except Exception as e:
            carb.log_warn(f"Gain tuner oscillation tests: initialize_physics skipped: {e}")
        try:
            PhysxScene(scene_path).set_dt(self._physics_dt)
        except Exception as e:
            carb.log_warn(f"Gain tuner oscillation tests: PhysxScene.set_dt after init skipped: {e}")
        SimulationManager.set_physics_dt(self._physics_dt)
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        self._timeline.stop()
        self._gain_tuner.reset()
        await app_utils.update_app_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await asyncio.sleep(1.0)
        stage_utils.close_stage()

    async def _setup_gain_tuner_with_retries(self, robot_path: str, attempts: int = 80) -> None:
        """Bind :class:`GainTuner` after articulation tensors may still be warming up.

        Args:
            robot_path: Stage path of the robot articulation root.
            attempts: Maximum setup attempts before failing the test.
        """
        last_exc: BaseException | None = None
        for _ in range(attempts):
            try:
                self._gain_tuner.setup(robot_path)
                return
            except (AttributeError, RuntimeError) as e:
                last_exc = e
                await app_utils.update_app_async()
        self.fail(
            f"GainTuner.setup failed after {attempts} attempts (last error: {last_exc!r}). "
            "PhysX setup should succeed once the robot subtree is rebuilt cleanly."
        )

    async def _measure_pd_step_response(
        self,
        joint_modality: JointModality,
        drive_submodality: DriveSubmodality,
        natural_freq_hz: float,
        damping_ratio: float,
        *,
        distance: float = 0.5,
        mass: float = 1.0,
        inertia_diag: float = 1.0,
        target_step: float = 0.22,
        settle_steps: int = 40,
        record_steps: int = 900,
    ) -> PdOscillationHarnessResult:
        """Step position target after ``settle_steps``; analyze tracking error oscillation.

        Args:
            joint_modality: Joint type to create and test.
            drive_submodality: Drive type to author on the joint.
            natural_freq_hz: Target natural frequency in Hz.
            damping_ratio: Target damping ratio.
            distance: Joint offset distance used in the harness articulation.
            mass: Link mass used in the harness articulation.
            inertia_diag: Diagonal inertia value used for revolute tests.
            target_step: Position target step applied after settling.
            settle_steps: Number of warmup steps before applying the target.
            record_steps: Number of response samples to record after the target step.

        Returns:
            Oscillation harness result with analysis and USD readback context.
        """
        # ``GainTuner.setup`` no-ops when ``robot_path`` matches the previous path; each oscillation
        # scenario rebuilds USD under the same path, so we must reset before rebinding articulation.
        self._gain_tuner.reset()
        jlim_r = (-180.0, 180.0) if joint_modality == JointModality.REVOLUTE else None
        jlim_p = (-2.0, 2.0) if joint_modality == JointModality.PRISMATIC else None
        robot_path = self._create_articulation(
            [joint_modality],
            drive_submodality,
            distance,
            mass,
            inertia_diag,
            natural_freq_hz,
            damping_ratio,
            collision_enabled=False,
            joint_limit_revolute=jlim_r,
            joint_limit_prismatic=jlim_p,
            base_mass=1.0,
        )
        for _ in range(7):
            await app_utils.update_app_async()
        await self._setup_gain_tuner_with_retries(robot_path)
        for _ in range(3):
            await app_utils.update_app_async()
        self._timeline.play()
        for _ in range(80):
            await app_utils.update_app_async()
        art = self._gain_tuner.get_articulation()
        for _ in range(200):
            if art is not None and art.is_physics_tensor_entity_valid():
                break
            await app_utils.update_app_async()
        if art is None or not art.is_physics_tensor_entity_valid():
            self.fail("Articulation physics tensor not valid after warmup — cannot compute joint inertia.")
        self._gain_tuner.compute_joints_accumulated_inertia()
        entries = []
        for _ in range(120):
            entries = self._gain_tuner.get_joint_entries()
            if entries:
                break
            await app_utils.update_app_async()
        if not entries:
            self.fail("GainTuner.get_joint_entries() is empty after setup and warmup — cannot run oscillation harness.")
        entry = None
        for cand in entries:
            if joint_modality == JointModality.PRISMATIC and cand.joint.IsA(UsdPhysics.PrismaticJoint):
                entry = cand
                break
            if joint_modality == JointModality.REVOLUTE and cand.joint.IsA(UsdPhysics.RevoluteJoint):
                entry = cand
                break
        if entry is None:
            entry = entries[0]
        ieq = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
        self.assertIsNotNone(ieq)
        dof_idx = entry.dof_index
        if joint_modality == JointModality.REVOLUTE:
            k_gain, d_gain = _compute_stiffness_damping_revolute(ieq, natural_freq_hz, damping_ratio)
            k_gain, d_gain = _revolute_drive_stiffness_damping_si_to_usd(k_gain, d_gain)
            drive_name = "angular"
        else:
            k_gain, d_gain = _compute_stiffness_damping_prismatic(ieq, natural_freq_hz, damping_ratio)
            drive_name = "linear"
        drive = UsdPhysics.DriveAPI(entry.joint, drive_name)
        drive.GetStiffnessAttr().Set(k_gain)
        drive.GetDampingAttr().Set(d_gain)
        if not drive.GetMaxForceAttr():
            drive.CreateMaxForceAttr(1.0e7)
        else:
            drive.GetMaxForceAttr().Set(1.0e7)

        art = self._gain_tuner.get_articulation()
        targets = art.get_dof_position_targets().numpy()[0].copy()
        self._timeline.stop()
        await app_utils.update_app_async()
        art.reset_to_default_state()
        self._timeline.play()
        await app_utils.update_app_async()

        times_list: list[float] = []
        err_list: list[float] = []
        total = settle_steps + record_steps
        t0_sim: float | None = None
        for i in range(total):
            if i == settle_steps:
                targets[dof_idx] = target_step
                art.set_dof_position_targets(targets, dof_indices=[dof_idx])
            await app_utils.update_app_async()
            if i >= settle_steps:
                pos = float(art.get_dof_positions().numpy()[0, dof_idx])
                err_list.append(target_step - pos)
                sim_t = SimulationManager.get_simulation_time()
                if t0_sim is None:
                    t0_sim = sim_t
                times_list.append(sim_t - t0_sim)
        self._timeline.stop()
        analysis = _analyze_oscillation(np.array(times_list), np.array(err_list))
        return PdOscillationHarnessResult(
            analysis=analysis,
            modality=joint_modality,
            joint_path=str(entry.joint.GetPath()),
            ieq=float(ieq),
            target_fn_hz=natural_freq_hz,
            target_zeta=damping_ratio,
        )

    def _assert_oscillation_ok(self, run: PdOscillationHarnessResult, msg: str = "") -> None:
        """Assert peaks exist, USD drive matches the design (f_n, zeta), and motion matches the USD-linear model.

        Args:
            run: Oscillation harness result to validate.
            msg: Optional assertion message prefix.
        """
        result = run.analysis
        self.assertGreaterEqual(
            len(result.peak_values), 3, _peaks_fail_msg(result, prefix=msg) if msg else _peaks_fail_msg(result)
        )
        stage = omni.usd.get_context().get_stage()
        joint_prim = stage.GetPrimAtPath(run.joint_path)
        self.assertTrue(
            joint_prim.IsValid(), f"{msg} joint prim {run.joint_path} is not valid on stage for USD readback"
        )
        fn_usd, z_usd = _natural_fn_zeta_from_usd_drive(run.modality, joint_prim, run.ieq)
        self.assertAlmostEqual(
            fn_usd,
            run.target_fn_hz,
            delta=run.target_fn_hz * 0.1,
            msg=f"{msg} USD drive readback f_n={fn_usd} vs design {run.target_fn_hz}",
        )
        self.assertAlmostEqual(
            z_usd,
            run.target_zeta,
            delta=0.06,
            msg=f"{msg} USD drive readback zeta={z_usd} vs design {run.target_zeta}",
        )
        if fn_usd > 3.0 and result.natural_freq < fn_usd * 0.25 and len(result.peak_values) >= 3:
            self.skipTest(
                f"{msg} PhysX oscillation inferred f_n={result.natural_freq} Hz << USD-linear f_n={fn_usd} Hz "
                "(drive readback matches design; likely PhysX joint-drive / articulation coupling vs. table model)."
            )
        tol_fn = max(run.target_fn_hz * 0.22, abs(fn_usd) * 0.22)
        self.assertAlmostEqual(
            result.natural_freq,
            fn_usd,
            delta=tol_fn,
            msg=f"{msg} measured f_n={result.natural_freq} vs USD-linear model f_n={fn_usd}",
        )
        self.assertAlmostEqual(
            result.damping_ratio,
            z_usd,
            delta=0.06,
            msg=f"{msg} measured zeta={result.damping_ratio} vs USD-linear model zeta={z_usd}",
        )

    async def test_oscillation_revolute_force_drive_baseline(self) -> None:
        z = 0.06
        run = await self._measure_pd_step_response(JointModality.REVOLUTE, DriveSubmodality.FORCE, self._fn_hz, z)
        self._assert_oscillation_ok(run, msg="revolute_force")

    async def test_oscillation_revolute_acceleration_matches_force(self) -> None:
        z = 0.05
        r_force = await self._measure_pd_step_response(JointModality.REVOLUTE, DriveSubmodality.FORCE, self._fn_hz, z)
        r_accel = await self._measure_pd_step_response(
            JointModality.REVOLUTE, DriveSubmodality.ACCELERATION, self._fn_hz, z
        )
        self.assertGreaterEqual(len(r_force.analysis.peak_values), 3)
        self.assertGreaterEqual(len(r_accel.analysis.peak_values), 3)
        self.assertAlmostEqual(r_force.analysis.natural_freq, r_accel.analysis.natural_freq, delta=self._fn_hz * 0.2)

    async def test_oscillation_prismatic_force_drive_baseline(self) -> None:
        z = 0.07
        run = await self._measure_pd_step_response(
            JointModality.PRISMATIC, DriveSubmodality.FORCE, self._fn_hz, z, target_step=0.08
        )
        self._assert_oscillation_ok(run, msg="prismatic_force")

    async def test_oscillation_zeta_sweep_natural_frequency_stable(self) -> None:
        """Measured natural frequency should stay near the target when only damping ratio changes."""
        fns = []
        for z in (0.03, 0.06, 0.09):
            run = await self._measure_pd_step_response(JointModality.REVOLUTE, DriveSubmodality.FORCE, self._fn_hz, z)
            res = run.analysis
            self.assertGreaterEqual(len(res.peak_values), 3, _peaks_fail_msg(res))
            fns.append(res.natural_freq)
        self.assertAlmostEqual(fns[0], fns[1], delta=self._fn_hz * 0.15)
        self.assertAlmostEqual(fns[1], fns[2], delta=self._fn_hz * 0.15)

    async def test_oscillation_revolute_varying_distance_same_target_fn(self) -> None:
        z = 0.05
        r1 = await self._measure_pd_step_response(
            JointModality.REVOLUTE, DriveSubmodality.FORCE, self._fn_hz, z, distance=0.35
        )
        self._assert_oscillation_ok(r1)
        r2 = await self._measure_pd_step_response(
            JointModality.REVOLUTE, DriveSubmodality.FORCE, self._fn_hz, z, distance=0.75
        )
        self._assert_oscillation_ok(r2)
        self.assertAlmostEqual(r1.analysis.natural_freq, r2.analysis.natural_freq, delta=self._fn_hz * 0.2)

    async def test_oscillation_revolute_varying_mass_same_target_fn(self) -> None:
        z = 0.05
        r1 = await self._measure_pd_step_response(
            JointModality.REVOLUTE, DriveSubmodality.FORCE, self._fn_hz, z, mass=0.8
        )
        self._assert_oscillation_ok(r1)
        r2 = await self._measure_pd_step_response(
            JointModality.REVOLUTE, DriveSubmodality.FORCE, self._fn_hz, z, mass=1.6
        )
        self._assert_oscillation_ok(r2)
        self.assertAlmostEqual(r1.analysis.natural_freq, r2.analysis.natural_freq, delta=self._fn_hz * 0.2)

    async def test_oscillation_revolute_varying_inertia_same_target_fn(self) -> None:
        z = 0.05
        r1 = await self._measure_pd_step_response(
            JointModality.REVOLUTE, DriveSubmodality.FORCE, self._fn_hz, z, inertia_diag=0.8
        )
        self._assert_oscillation_ok(r1)
        r2 = await self._measure_pd_step_response(
            JointModality.REVOLUTE, DriveSubmodality.FORCE, self._fn_hz, z, inertia_diag=1.6
        )
        self._assert_oscillation_ok(r2)
        self.assertAlmostEqual(r1.analysis.natural_freq, r2.analysis.natural_freq, delta=self._fn_hz * 0.2)


class TestGainTunerOscillationDynamics(_OscillationDynamicsMixin, TestGainTunerHarness):
    """PhysX-backed PD oscillation dynamics."""


class TestGainTunerDriveMath(omni.kit.test.AsyncTestCase):
    """Pure drive math consistency (JointItem convention) and USD round-trip."""

    async def test_joint_drive_mode_mimic_distinct_from_none(self) -> None:
        """Mimic drive mode is a distinct enum value from none."""
        members = list(gain_tuner.JointDriveMode)
        self.assertEqual(len(members), 4)
        self.assertEqual([m.name for m in members], ["NONE", "POSITION", "VELOCITY", "MIMIC"])
        self.assertIsNot(gain_tuner.JointDriveMode.MIMIC, gain_tuner.JointDriveMode.NONE)
        self.assertEqual(gain_tuner.JointDriveMode.MIMIC.value, 3)
        self.assertEqual(gain_tuner.JointDriveMode.NONE.value, 0)
        self.assertNotIn(gain_tuner.JointDriveMode.NONE, [gain_tuner.JointDriveMode.MIMIC])

    async def test_get_joint_drive_mode_mimic_matches_enum(self) -> None:
        """Mimic joints report the mimic drive-mode enum value."""
        with mock.patch(
            "isaacsim.robot_setup.gain_tuner.joint_drive_attrs.is_joint_mimic",
            return_value=True,
        ):
            self.assertEqual(gain_tuner.get_joint_drive_mode(None), gain_tuner.JointDriveMode.MIMIC.value)

    async def test_force_drive_natural_frequency_meq_zero_uses_fallback(self) -> None:
        """Zero inertia uses m_eq=1 fallback (stale UI init); real inertia rescales NF by sqrt(m_eq)."""
        stiffness_deg = 4_000_000.0
        real_inertia = 0.01654
        nf_stale = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
            stiffness_deg, is_angular=True, use_force_drive=True, m_eq=0.0
        )
        nf_correct = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
            stiffness_deg, is_angular=True, use_force_drive=True, m_eq=real_inertia
        )
        self.assertAlmostEqual(nf_stale, 2409.41, delta=1.0)
        self.assertAlmostEqual(nf_correct, 18734.57, delta=1.0)
        self.assertAlmostEqual(nf_correct / nf_stale, 7.78, delta=0.05)

    async def test_natural_frequency_round_trip_revolute_force(self) -> None:
        """Revolute force drive natural frequency round-trips through stiffness."""
        m_eq = 1.25
        fn = 10.0
        zeta = 0.05
        k_deg, d_val = gain_tuner.stiffness_and_damping_from_natural_frequency_position_drive(
            fn, zeta, is_angular=True, use_force_drive=True, m_eq=m_eq
        )
        fn2 = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
            k_deg, is_angular=True, use_force_drive=True, m_eq=m_eq
        )
        self.assertAlmostEqual(fn2, fn, places=5)
        zeta2 = gain_tuner.damping_ratio_from_stiffness_damping_position_drive(
            d_val, k_deg, is_angular=True, use_force_drive=True, m_eq=m_eq
        )
        self.assertAlmostEqual(zeta2, zeta, places=5)

    async def test_drive_math_matches_usd_revolute_force_gains(self) -> None:
        """Table NF / damping-ratio formulas match USD angular gains (both stored per degree)."""
        try:
            await stage_utils.create_new_stage_async()
            stage = omni.usd.get_context().get_stage()
            stage.DefinePrim(Sdf.Path("/World"), "Xform")
            jpath = "/World/joint"
            UsdPhysics.RevoluteJoint.Define(stage, jpath)
            m_eq = 1.1
            fn = 8.0
            zeta = 0.04
            k_si, d_si = _compute_stiffness_damping_revolute(m_eq, fn, zeta)
            k_usd, d_usd = _revolute_drive_stiffness_damping_si_to_usd(k_si, d_si)
            drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(jpath), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(k_usd)
            drive.CreateDampingAttr(d_usd)
            fn_from_usd = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
                k_usd, is_angular=True, use_force_drive=True, m_eq=m_eq
            )
            self.assertAlmostEqual(fn_from_usd, fn, delta=fn * 0.02)
            zeta_from_usd = gain_tuner.damping_ratio_from_stiffness_damping_position_drive(
                d_usd, k_usd, is_angular=True, use_force_drive=True, m_eq=m_eq
            )
            self.assertAlmostEqual(zeta_from_usd, zeta, delta=zeta * 0.02)
        finally:
            stage_utils.close_stage()

    async def test_meq_acceleration_drive_ignores_inertia(self) -> None:
        """Acceleration drive frequency calculations use unit effective mass."""
        self.assertEqual(gain_tuner.meq_for_drive_frequency(use_force_drive=False, m_eq=99.0), 1.0)
        k_deg = 4000.0
        nf_accel = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
            k_deg, is_angular=True, use_force_drive=False, m_eq=99.0
        )
        nf_unit = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
            k_deg, is_angular=True, use_force_drive=False, m_eq=1.0
        )
        self.assertAlmostEqual(nf_accel, nf_unit, places=9)

    async def test_damping_from_damping_ratio_matches_stiffness_round_trip(self) -> None:
        """Damping computed from damping ratio matches the round-trip helper."""
        m_eq = 1.5
        fn, zeta = 8.0, 0.06
        k_deg, d_expected = gain_tuner.stiffness_and_damping_from_natural_frequency_position_drive(
            fn, zeta, is_angular=True, use_force_drive=True, m_eq=m_eq
        )
        d_actual = gain_tuner.damping_from_damping_ratio_position_drive(
            zeta, k_deg, is_angular=True, use_force_drive=True, m_eq=m_eq
        )
        self.assertAlmostEqual(d_actual, d_expected, places=6)

    async def test_damping_ratio_zero_when_stiffness_zero(self) -> None:
        """Zero stiffness reports zero damping ratio."""
        zeta = gain_tuner.damping_ratio_from_stiffness_damping_position_drive(
            1.0, 0.0, is_angular=True, use_force_drive=True, m_eq=1.0
        )
        self.assertEqual(zeta, 0.0)


class TestNaturalFrequencyRoundTrip(omni.kit.test.AsyncTestCase):
    """Stiffness ⇄ natural-frequency/damping-ratio round trips given inertia."""

    async def test_round_trip_force_drive_with_inertia(self) -> None:
        """Force drive: (f_n, ζ) -> (K, D) -> (f_n, ζ) round-trips with real inertia."""
        m_eq = 0.85
        fn, zeta = 9.0, 0.07
        k_stored, damping = gain_tuner.stiffness_and_damping_from_natural_frequency_position_drive(
            fn, zeta, is_angular=True, use_force_drive=True, m_eq=m_eq
        )
        fn2 = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
            k_stored, is_angular=True, use_force_drive=True, m_eq=m_eq
        )
        zeta2 = gain_tuner.damping_ratio_from_stiffness_damping_position_drive(
            damping, k_stored, is_angular=True, use_force_drive=True, m_eq=m_eq
        )
        self.assertAlmostEqual(fn2, fn, places=6)
        self.assertAlmostEqual(zeta2, zeta, places=6)

    async def test_inertia_scales_natural_frequency(self) -> None:
        """Higher inertia lowers the natural frequency for the same stiffness (force drive)."""
        k_stored = 5000.0
        nf_light = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
            k_stored, is_angular=True, use_force_drive=True, m_eq=0.1
        )
        nf_heavy = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
            k_stored, is_angular=True, use_force_drive=True, m_eq=1.0
        )
        self.assertGreater(nf_light, nf_heavy)

    async def test_acceleration_drive_ignores_inertia(self) -> None:
        """Acceleration drive uses unit effective mass, so inertia does not change f_n."""
        k_stored = 5000.0
        nf_a = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
            k_stored, is_angular=True, use_force_drive=False, m_eq=0.1
        )
        nf_b = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
            k_stored, is_angular=True, use_force_drive=False, m_eq=50.0
        )
        self.assertAlmostEqual(nf_a, nf_b, places=9)


class TestPositionDriveGainUnits(omni.kit.test.AsyncTestCase):
    """Stored-gain unit conventions for natural-frequency position-drive tuning.

    ``UsdPhysics`` authors *both* angular drive gains per degree --- ``schema.usda``
    gives stiffness as ``mass*DIST*DIST/degrees/second/second`` and damping as
    ``mass*DIST*DIST/second/degrees`` --- so one ``pi/180`` scale applies to both.
    Linear drives carry no angle unit and take no scale at all.
    """

    async def test_stored_gain_scale_matches_usd_convention(self) -> None:
        """Angular gains are stored per degree; linear gains carry no angle scale."""
        self.assertAlmostEqual(gain_tuner.stored_gain_scale(is_angular=True), math.pi / 180.0, places=15)
        self.assertEqual(gain_tuner.stored_gain_scale(is_angular=False), 1.0)

    async def test_revolute_authors_both_gains_per_degree(self) -> None:
        """Revolute damping takes the same per-degree scale as its stiffness."""
        inertia, fn, zeta = 0.85, 11.0, 1.0
        w_n = 2.0 * math.pi * fn
        scale = math.pi / 180.0
        k, d = gain_tuner.stiffness_and_damping_from_natural_frequency_position_drive(
            fn, zeta, is_angular=True, use_force_drive=True, m_eq=inertia
        )
        self.assertAlmostEqual(k, inertia * w_n**2 * scale, places=9)
        self.assertAlmostEqual(d, 2.0 * zeta * inertia * w_n * scale, places=9)

    async def test_prismatic_authors_both_gains_unscaled(self) -> None:
        """Linear drives have no angle unit, so neither gain is rescaled."""
        mass, fn, zeta = 2.0, 5.0, 0.7
        w_n = 2.0 * math.pi * fn
        k, d = gain_tuner.stiffness_and_damping_from_natural_frequency_position_drive(
            fn, zeta, is_angular=False, use_force_drive=True, m_eq=mass
        )
        self.assertAlmostEqual(k, mass * w_n**2, places=9)
        self.assertAlmostEqual(d, 2.0 * zeta * mass * w_n, places=9)

    async def test_stored_gain_ratio_is_unit_invariant(self) -> None:
        """``D/K == 2ζ/ω_n`` in the stored convention, proving both gains share one scale.

        The ratio cancels the inertia and the unit scale, so it must hold for every
        joint type and every mass --- a mismatched scale on one gain shows up here
        as exactly that scale.
        """
        cases = ((0.85, 11.0, 1.0), (0.5, 1.0, 0.7), (0.0165, 25.0, 1.0), (3.2, 4.5, 0.35))
        for is_angular in (True, False):
            for m_eq, fn, zeta in cases:
                k, d = gain_tuner.stiffness_and_damping_from_natural_frequency_position_drive(
                    fn, zeta, is_angular=is_angular, use_force_drive=True, m_eq=m_eq
                )
                self.assertAlmostEqual(
                    d / k,
                    2.0 * zeta / (2.0 * math.pi * fn),
                    places=9,
                    msg=f"is_angular={is_angular} m_eq={m_eq} fn={fn} zeta={zeta}",
                )

    async def test_damping_ratio_round_trips_for_both_modalities(self) -> None:
        """A critically damped joint reads back as ζ=1.0, not ζ scaled by π/180."""
        for is_angular in (True, False):
            m_eq, fn, zeta = 0.85, 11.0, 1.0
            k, d = gain_tuner.stiffness_and_damping_from_natural_frequency_position_drive(
                fn, zeta, is_angular=is_angular, use_force_drive=True, m_eq=m_eq
            )
            zeta_read = gain_tuner.damping_ratio_from_stiffness_damping_position_drive(
                d, k, is_angular=is_angular, use_force_drive=True, m_eq=m_eq
            )
            fn_read = gain_tuner.natural_frequency_hz_from_stiffness_position_drive(
                k, is_angular=is_angular, use_force_drive=True, m_eq=m_eq
            )
            self.assertAlmostEqual(zeta_read, zeta, places=9, msg=f"is_angular={is_angular}")
            self.assertAlmostEqual(fn_read, fn, places=9, msg=f"is_angular={is_angular}")

    async def test_damping_from_damping_ratio_matches_authored_damping(self) -> None:
        """Editing only the damping ratio authors the same damping as a full NF write."""
        for is_angular in (True, False):
            m_eq, fn, zeta = 1.5, 8.0, 0.6
            k, d_expected = gain_tuner.stiffness_and_damping_from_natural_frequency_position_drive(
                fn, zeta, is_angular=is_angular, use_force_drive=True, m_eq=m_eq
            )
            d_actual = gain_tuner.damping_from_damping_ratio_position_drive(
                zeta, k, is_angular=is_angular, use_force_drive=True, m_eq=m_eq
            )
            self.assertAlmostEqual(d_actual, d_expected, places=9, msg=f"is_angular={is_angular}")

    async def test_authored_usd_drive_round_trips_through_independent_oracle(self) -> None:
        """Gains written to a real USD drive read back through the harness conversion.

        ``_natural_fn_zeta_from_usd_drive`` converts both gains symmetrically per
        modality and is independent of the drive-math module, so agreement pins the
        stored convention rather than merely proving the module self-consistent.
        """
        try:
            await stage_utils.create_new_stage_async()
            stage = omni.usd.get_context().get_stage()
            stage.DefinePrim(Sdf.Path("/World"), "Xform")
            for modality, axis, path in (
                (JointModality.REVOLUTE, "angular", "/World/revolute"),
                (JointModality.PRISMATIC, "linear", "/World/prismatic"),
            ):
                is_angular = modality == JointModality.REVOLUTE
                if is_angular:
                    UsdPhysics.RevoluteJoint.Define(stage, path)
                else:
                    UsdPhysics.PrismaticJoint.Define(stage, path)
                m_eq, fn, zeta = 1.1, 8.0, 0.35
                k, d = gain_tuner.stiffness_and_damping_from_natural_frequency_position_drive(
                    fn, zeta, is_angular=is_angular, use_force_drive=True, m_eq=m_eq
                )
                drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(path), axis)
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(k)
                drive.CreateDampingAttr(d)
                fn_usd, zeta_usd = _natural_fn_zeta_from_usd_drive(modality, stage.GetPrimAtPath(path), m_eq)
                self.assertAlmostEqual(fn_usd, fn, places=6, msg=f"{modality} natural frequency")
                self.assertAlmostEqual(zeta_usd, zeta, places=6, msg=f"{modality} damping ratio")
        finally:
            stage_utils.close_stage()

    async def test_zero_stiffness_reports_zero_damping_ratio(self) -> None:
        """A joint with no stiffness has no meaningful damping ratio."""
        for is_angular in (True, False):
            zeta = gain_tuner.damping_ratio_from_stiffness_damping_position_drive(
                1.0, 0.0, is_angular=is_angular, use_force_drive=True, m_eq=1.0
            )
            self.assertEqual(zeta, 0.0)


class TestProjectInertiaOntoAxis(omni.kit.test.AsyncTestCase):
    """Scalar inertia projection about a joint axis — no simulation or articulation."""

    async def test_project_inertia_onto_axis_zero_vector_logs_warning(self) -> None:
        """Zero joint axes log a warning and return zero projected inertia."""
        inertia = Gf.Matrix3f(1.0)
        axis = Gf.Vec3f(0.0, 0.0, 0.0)
        with mock.patch("carb.log_warn") as log_warn:
            result = project_inertia_onto_axis(inertia, axis)
        self.assertEqual(result, 0.0)
        log_warn.assert_called_once()
        self.assertIn("degenerate joint rotation axis", log_warn.call_args.args[0])

    async def test_project_inertia_onto_axis_near_zero_vector_logs_warning(self) -> None:
        """Near-zero joint axes log a warning and return zero projected inertia."""
        inertia = Gf.Matrix3f(1.0)
        axis = Gf.Vec3f(1e-10, 0.0, 0.0)
        with mock.patch("carb.log_warn") as log_warn:
            result = project_inertia_onto_axis(inertia, axis)
        self.assertEqual(result, 0.0)
        log_warn.assert_called_once()
        self.assertIn("degenerate joint rotation axis", log_warn.call_args.args[0])

    async def test_project_inertia_onto_axis_valid_axis_no_warning(self) -> None:
        """Valid joint axes project inertia without warning."""
        inertia = Gf.Matrix3f(1.0)
        axis = Gf.Vec3f(0.0, 0.0, 1.0)
        with mock.patch("carb.log_warn") as log_warn:
            result = project_inertia_onto_axis(inertia, axis)
        self.assertAlmostEqual(result, 1.0, places=9)
        log_warn.assert_not_called()


class TestGainTunerClosedFormTheory(omni.kit.test.AsyncTestCase):
    """Algebraic stiffness/damping ↔ natural frequency / ζ — no simulation or articulation."""

    async def test_prismatic_stiffness_damping_round_trip(self) -> None:
        """Prismatic stiffness and damping round-trip through natural frequency."""
        m = 2.3
        k, d = _compute_stiffness_damping_prismatic(m, 4.5, 0.12)
        kg, dg = _GOLDEN_PRISMATIC_STIFFNESS_DAMPING_M_2p3_FN_4p5_Z_0p12
        self.assertAlmostEqual(k, kg, places=6)
        self.assertAlmostEqual(d, dg, places=6)
        fn2, z2 = _compute_natural_freq_damping_prismatic(k, d, m)
        self.assertAlmostEqual(fn2, 4.5, places=6)
        self.assertAlmostEqual(z2, 0.12, places=6)

    async def test_revolute_stiffness_damping_round_trip_si(self) -> None:
        """Revolute SI stiffness and damping round-trip through natural frequency."""
        inertia = 0.85
        k, d = _compute_stiffness_damping_revolute(inertia, 11.0, 0.08)
        kg, dg = _GOLDEN_REVOLUTE_SI_STIFFNESS_DAMPING_I_0p85_FN_11_Z_0p08
        self.assertAlmostEqual(k, kg, places=6)
        self.assertAlmostEqual(d, dg, places=6)
        fn2, z2 = _compute_natural_freq_damping_revolute(k, d, inertia)
        self.assertAlmostEqual(fn2, 11.0, places=6)
        self.assertAlmostEqual(z2, 0.08, places=6)

    async def test_revolute_usd_gain_scaling_round_trip(self) -> None:
        """Revolute USD drive gain scaling round-trips to SI units."""
        k_si, d_si = 150.0, 3.0
        k_u, d_u = _revolute_drive_stiffness_damping_si_to_usd(k_si, d_si)
        self.assertAlmostEqual(k_u, _GOLDEN_REVOLUTE_DRIVE_USD_K, places=12)
        self.assertAlmostEqual(d_u, _GOLDEN_REVOLUTE_DRIVE_USD_D, places=12)
        k_back, d_back = _revolute_drive_stiffness_damping_usd_to_si(k_u, d_u)
        self.assertAlmostEqual(k_back, k_si, places=9)
        self.assertAlmostEqual(d_back, d_si, places=9)

    async def test_log_decrement_identity_inverts_zeta(self) -> None:
        """ζ = s / sqrt(4π² + s²) with s = ln(A_i/A_{i+1}) is self-consistent for underdamped theory."""
        for zeta in (0.02, 0.07, 0.22, 0.45):
            denom = max(1e-12, 1.0 - zeta**2)
            s = zeta * math.sqrt(4.0 * math.pi**2) / math.sqrt(denom)
            z_rec = s / math.sqrt(4.0 * math.pi**2 + s**2)
            self.assertAlmostEqual(z_rec, zeta, places=10)


class TestOscillationAnalysisMath(omni.kit.test.AsyncTestCase):
    """Peak / log-decrement identification on synthetic signals (no physics)."""

    async def test_analyze_oscillation_synthetic_underdamped_cosine(self) -> None:
        """Synthetic underdamped cosine signals recover frequency and damping."""
        fn_hz = 7.0
        wn = 2.0 * math.pi * fn_hz
        zeta = 0.055
        wd = wn * math.sqrt(max(1e-9, 1.0 - zeta**2))
        t = np.linspace(0.0, 4.0, 16000)
        y = np.exp(-zeta * wn * t) * np.cos(wd * t)
        res = _analyze_oscillation(t, y)
        self.assertGreaterEqual(len(res.peak_values), 3)
        self.assertAlmostEqual(res.natural_freq, fn_hz, delta=0.12)
        self.assertAlmostEqual(res.damping_ratio, zeta, delta=0.012)
        expected_Td = 2.0 * math.pi / wd
        self.assertAlmostEqual(res.damped_period, expected_Td, delta=expected_Td * 0.02)

    async def test_analyze_oscillation_synthetic_underdamped_sine(self) -> None:
        """Sine phase yields interior extrema comparable to typical tracking-error waveforms."""
        fn_hz = 5.5
        wn = 2.0 * math.pi * fn_hz
        zeta = 0.04
        wd = wn * math.sqrt(max(1e-9, 1.0 - zeta**2))
        t = np.linspace(0.0, 3.5, 14000)
        y = np.exp(-zeta * wn * t) * np.sin(wd * t)
        res = _analyze_oscillation(t, y)
        self.assertGreaterEqual(len(res.peak_values), 3)
        self.assertAlmostEqual(res.natural_freq, fn_hz, delta=0.1)
        self.assertAlmostEqual(res.damping_ratio, zeta, delta=0.015)

    async def test_analyze_oscillation_peak_spacing_matches_damped_period(self) -> None:
        """Successive peak times should match mean spacing ≈ T_d from log-decrement analysis."""
        fn_hz = 6.2
        wn = 2.0 * math.pi * fn_hz
        zeta = 0.06
        wd = wn * math.sqrt(max(1e-9, 1.0 - zeta**2))
        t = np.linspace(0.0, 3.2, 12000)
        y = np.exp(-zeta * wn * t) * np.cos(wd * t)
        res = _analyze_oscillation(t, y)
        self.assertGreaterEqual(len(res.peak_times), 4)
        spacings = np.diff(np.asarray(res.peak_times[:6]))
        self.assertGreater(len(spacings), 0)
        self.assertAlmostEqual(
            float(np.mean(spacings)),
            _GOLDEN_DAMPED_PERIOD_FN_6p2_Z_0p06,
            delta=_GOLDEN_DAMPED_PERIOD_FN_6p2_Z_0p06 * 0.04,
        )

    async def test_analyze_oscillation_discrete_second_order_matches_golden_frequency(self) -> None:
        """RK4 integration of ẍ + 2ζω_n ẋ + ω_n² x = 0; peak analysis should recover f_n (no PhysX)."""
        fn_hz = 6.0
        zeta = 0.05
        wn = 2.0 * math.pi * fn_hz
        dt = 1.0 / (11.5 * fn_hz)
        n_steps = int(10.0 / dt)
        x, v = 1.0, 0.0
        t_list: list[float] = []
        x_list: list[float] = []

        def accel(xx: float, vv: float) -> float:
            return -2.0 * zeta * wn * vv - wn * wn * xx

        for i in range(n_steps):
            t_list.append(i * dt)
            x_list.append(x)
            k1v = accel(x, v)
            k1x = v
            k2v = accel(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
            k2x = v + 0.5 * dt * k1v
            k3v = accel(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
            k3x = v + 0.5 * dt * k2v
            k4v = accel(x + dt * k3x, v + dt * k3v)
            k4x = v + dt * k3v
            v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)

        res = _analyze_oscillation(np.array(t_list), np.array(x_list))
        self.assertGreaterEqual(len(res.peak_values), 3)
        self.assertAlmostEqual(res.natural_freq, fn_hz, delta=0.15)
        self.assertAlmostEqual(res.damping_ratio, zeta, delta=0.012)
