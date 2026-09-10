# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""dt physics sweep: characterize how position-control accuracy degrades as the physics timestep coarsens.

The sweep runs a single step-command accuracy probe (:class:`DiscretizationSweepTest`)
at each of several physics timesteps and compares the per-joint settling time and
steady-state error across timesteps.  The single-level probe lives here; the caller
(the UI layer) changes the physics dt between invocations and feeds the collected
per-level results into :func:`aggregate_dt_sweep` to classify each joint at a target dt.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from math import log10

import numpy as np
from isaacsim.core.experimental.prims import Articulation

from .base import RobotTest, TestResult

_VELOCITY_THRESHOLD = 0.01
_SETTLE_WINDOW = 0.25

# Joint control-mode codes carried in the ``joint_modes`` mapping (mirror
# ``gains_tuner.JointMode``): a DOF index maps to POSITION (0), VELOCITY (1), or
# NONE (2). The dt sweep only commands position-mode DOFs; an unlisted DOF defaults
# to NONE so it is excluded.
_JOINT_MODE_POSITION = 0
_JOINT_MODE_NONE = 2


def validate_dt_bounds(dt_max: float, dt_min: float) -> None:
    """Validate dt-sweep bounds, raising ``ValueError`` on unusable input.

    The sweep sweeps from the coarsest timestep (``dt_max``) down to the finest
    (``dt_min``) and aggregation assumes that ordering, so bounds that are
    reversed, equal, non-positive, or non-finite would silently produce a wrong
    accuracy cliff / verdict.  Reject them up front.

    Args:
        dt_max: Coarsest physics timestep in seconds (must be > ``dt_min`` > 0).
        dt_min: Finest physics timestep in seconds (must be > 0 and < ``dt_max``).

    Raises:
        ValueError: If either bound is non-finite or non-positive, or if
            ``dt_max`` is not strictly greater than ``dt_min``.
    """
    for name, value in (("dt Max", dt_max), ("dt Min", dt_min)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number (got {value!r}).")
        if value <= 0.0:
            raise ValueError(f"{name} must be positive (got {value!r}).")
    if dt_max <= dt_min:
        raise ValueError(
            f"dt Max ({dt_max!r}) must be strictly greater than dt Min ({dt_min!r}); "
            "the sweep runs from the coarsest timestep down to the finest."
        )


def dt_sweep_levels(dt_max: float, dt_min: float, num_steps: int) -> np.ndarray:
    """Compute the logarithmically-spaced physics timesteps for a dt sweep.

    The levels run from the coarsest timestep (``dt_max``) to the finest
    (``dt_min``) so degradation is measured relative to the finest reference.

    Args:
        dt_max: Coarsest physics timestep in seconds (swept first).
        dt_min: Finest physics timestep in seconds (swept last, used as the reference).
        num_steps: Number of timesteps to sample.

    Returns:
        Array of timesteps from ``dt_max`` down to ``dt_min``.  Returns a single
        clamped value when ``num_steps`` is 1 and an empty array when it is < 1.

    Raises:
        ValueError: If the bounds are non-finite, non-positive, or reversed/equal
            (see :func:`validate_dt_bounds`).  For a single-level sweep only the
            finite/positive check on ``dt_max`` applies.
    """
    num_steps = int(num_steps)
    if num_steps < 1:
        return np.empty(0)
    dt_max = float(dt_max)
    dt_min = float(dt_min)
    if num_steps == 1:
        if not math.isfinite(dt_max) or dt_max <= 0.0:
            raise ValueError(f"dt Max must be a finite positive number (got {dt_max!r}).")
        return np.array([dt_max])
    validate_dt_bounds(dt_max, dt_min)
    return np.logspace(log10(dt_max), log10(dt_min), num_steps)


def select_target_level_index(dt_levels, target_dt: float) -> int:
    """Return the index of the sweep level that represents the user's target dt.

    Works on any ordered (coarse-to-fine, as produced by :func:`dt_sweep_levels`)
    sequence of timesteps, so callers can select the target level either from the
    raw ``dt_levels`` array or from a list of per-level ``dt`` values (recorded
    trajectories, effort history, aggregated sweep levels).

    Args:
        dt_levels: Ordered (coarse-to-fine) physics timesteps for the sweep.
        target_dt: The physics timestep the user intends to simulate at.

    Returns:
        The index of the first (coarsest) level whose timestep is at or below
        ``target_dt``, falling back to the finest (last) level, or ``-1`` when
        ``dt_levels`` is empty.
    """
    n = len(dt_levels)
    if n == 0:
        return -1
    for i in range(n):
        if float(dt_levels[i]) <= float(target_dt) + 1e-9:
            return i
    return n - 1


def _compute_level_degradations(sweep: list[dict], reference: dict | None, tolerance: float) -> None:
    """Fill in ``settle_degradation`` / ``error_degradation`` for each sweep level in place.

    Degradation is the fractional increase of a level's settling time (or
    steady-state error) relative to the finest-dt ``reference`` level.  Levels are
    left at ``None`` when the reference is missing, the level was skipped, or the
    reference metric is too small to divide by.

    Args:
        sweep: Per-level result dicts for a single joint (mutated in place).
        reference: The finest-dt reference level, or None when unavailable.
        tolerance: Position error threshold below which errors count as accurate.
    """
    for level in sweep:
        level["settle_degradation"] = None
        level["error_degradation"] = None
        if level.get("skipped") or reference is None:
            continue

        ref_settle = reference.get("settling_time")
        level_settle = level.get("settling_time")
        if ref_settle is not None and ref_settle > 1e-9 and level_settle is not None:
            level["settle_degradation"] = (level_settle - ref_settle) / ref_settle

        ref_error = reference.get("steady_state_error", float("nan"))
        level_error = level.get("steady_state_error", float("nan"))
        if math.isfinite(ref_error) and math.isfinite(level_error):  # reject NaN and +/-inf
            if ref_error < tolerance and level_error < tolerance:
                level["error_degradation"] = 0.0
            elif ref_error > 1e-12:
                level["error_degradation"] = (level_error - ref_error) / ref_error


def _level_is_absolutely_accurate(level: dict, tolerance: float) -> bool:
    """Return True when a level settled and its absolute steady-state error is within tolerance.

    Args:
        level: A per-level result dict with ``settled`` / ``steady_state_error``.
        tolerance: Absolute steady-state position-error threshold.

    Returns:
        True only when the level settled and reports a finite steady-state error
        at or below ``tolerance``.
    """
    if not level.get("settled"):
        return False
    ss_error = level.get("steady_state_error", float("nan"))
    return ss_error is not None and math.isfinite(ss_error) and ss_error <= tolerance


def _level_is_accurate(level: dict, settle_threshold: float, error_threshold: float, tolerance: float) -> bool:
    """Return True when a level is accurate at its timestep.

    A level counts as accurate only when it settled, its own absolute steady-state
    error is within ``tolerance``, and its settle- and error-degradation relative to
    the finest reference are both within threshold.  Requiring the absolute check on
    every level (including the reference, whose self-relative degradation is trivially
    zero) prevents a level that merely tracks an inaccurate reference from being
    reported as accurate.

    Args:
        level: A per-level result dict with degradation fields already populated.
        settle_threshold: Maximum allowed fractional settling-time degradation.
        error_threshold: Maximum allowed fractional steady-state-error degradation.
        tolerance: Absolute steady-state position-error threshold.

    Returns:
        True only when the level is both absolutely accurate and within both
        degradation thresholds.
    """
    if not _level_is_absolutely_accurate(level, tolerance):
        return False
    settle_deg = level.get("settle_degradation")
    error_deg = level.get("error_degradation")
    return (
        settle_deg is not None
        and settle_deg <= settle_threshold
        and error_deg is not None
        and error_deg <= error_threshold
    )


def _find_accuracy_cliff(
    sweep: list[dict], settle_threshold: float, error_threshold: float, tolerance: float
) -> float | None:
    """Find the coarsest timestep whose accuracy is still within threshold.

    Walking from coarse to fine, this returns the first (coarsest) level that is
    accurate (see :func:`_level_is_accurate`).  That timestep is the "accuracy
    cliff": the largest dt that is still absolutely accurate and tracks the
    finest-dt reference.  When the finest sampled level is itself inaccurate no
    level qualifies and None is returned.

    Args:
        sweep: Per-level result dicts for a single joint (coarse-to-fine order).
        settle_threshold: Maximum allowed fractional settling-time degradation.
        error_threshold: Maximum allowed fractional steady-state-error degradation.
        tolerance: Absolute steady-state error threshold.

    Returns:
        The coarsest acceptable timestep, or None when no level qualifies.
    """
    for level in sweep:
        if level.get("skipped"):
            continue
        if _level_is_accurate(level, settle_threshold, error_threshold, tolerance):
            return level["dt"]
    return None


def _classify_target_level(
    target_level: dict | None, settle_threshold: float, error_threshold: float, tolerance: float
) -> str:
    """Classify a joint's accuracy at the target timestep.

    Args:
        target_level: The sweep level nearest the target dt, or None.
        settle_threshold: Maximum allowed fractional settling-time degradation.
        error_threshold: Maximum allowed fractional steady-state-error degradation.
        tolerance: Absolute steady-state error threshold.

    Returns:
        ``"did_not_settle"`` when the joint never settled at the target dt,
        ``"accurate"`` when the level is absolutely accurate and within both
        degradation thresholds, otherwise ``"degraded"``.
    """
    if target_level is None or not target_level.get("settled"):
        return "did_not_settle"
    if _level_is_accurate(target_level, settle_threshold, error_threshold, tolerance):
        return "accurate"
    return "degraded"


def aggregate_dt_sweep(
    per_joint_sweep: dict[int, list[dict]],
    target_dt: float,
    settle_threshold: float,
    error_threshold: float,
    tolerance: float,
) -> dict[int, dict]:
    """Aggregate per-level dt-sweep samples into per-joint accuracy metrics.

    For each joint the finest non-skipped timestep is used as the accuracy
    reference; every level's settling time and steady-state error are compared to
    it to compute degradation percentages, the accuracy cliff (coarsest still-accurate
    timestep), and a classification at the user's target dt.

    Args:
        per_joint_sweep: Map of DOF index -> ordered (coarse-to-fine) list of
            per-level result dicts, each with ``dt`` / ``settling_time`` /
            ``steady_state_error`` / ``settled`` / ``skipped`` keys.
        target_dt: The physics timestep the user intends to simulate at.
        settle_threshold: Maximum allowed fractional settling-time degradation.
        error_threshold: Maximum allowed fractional steady-state-error degradation.
        tolerance: Position error threshold below which errors count as accurate.

    Returns:
        Map of DOF index -> combined metrics dict carrying ``status``,
        ``accuracy_cliff_dt``, ``target_dt``, ``target_level``, the annotated
        ``dt_sweep`` levels, and the thresholds used.

    Raises:
        ValueError: If ``target_dt`` is non-finite or non-positive.
    """
    if not math.isfinite(target_dt) or target_dt <= 0.0:
        raise ValueError(f"target_dt must be a finite positive number (got {target_dt!r}).")

    combined_metrics: dict[int, dict] = {}
    for dof_idx, sweep in per_joint_sweep.items():
        non_skipped = [level for level in sweep if not level.get("skipped")]
        reference = non_skipped[-1] if non_skipped else None

        _compute_level_degradations(sweep, reference, tolerance)
        # Pick the level nearest (and no finer than) the target dt via the shared
        # public selector.  Restricting to non-skipped levels (already coarse-to-fine)
        # preserves the "first at/below target, else finest" semantics.
        target_idx = select_target_level_index([level["dt"] for level in non_skipped], target_dt)
        target_level = non_skipped[target_idx] if target_idx >= 0 else None

        # A finest-dt reference that ran but never settled gives no valid baseline,
        # so per-level degradations are meaningless: report the joint as
        # indeterminate rather than mislabeling a settled target dt as degraded.
        # (An all-skipped joint has no reference at all and falls through to the
        # normal path, which classifies it as did_not_settle.)
        if reference is not None and not reference.get("settled"):
            accuracy_cliff_dt = None
            classification = "indeterminate"
        else:
            accuracy_cliff_dt = _find_accuracy_cliff(sweep, settle_threshold, error_threshold, tolerance)
            classification = _classify_target_level(target_level, settle_threshold, error_threshold, tolerance)

        combined_metrics[dof_idx] = {
            "test_type": "discretization_sweep",
            "status": classification,
            "accuracy_cliff_dt": accuracy_cliff_dt,
            "target_dt": target_dt,
            "target_level": target_level,
            "dt_sweep": sweep,
            "settle_threshold": settle_threshold,
            "error_threshold": error_threshold,
        }
    return combined_metrics


class DiscretizationSweepTest(RobotTest):
    """Single-timestep accuracy probe used by the dt physics sweep.

    Commands the active position-mode joints to a fixed step target (the center
    of each joint's range offset by a configurable fraction of the half-range)
    and measures the per-joint settling time and steady-state error over an
    approach + hold cycle.  The settle condition mirrors the Snap-to-Limits test:
    velocity below ``_VELOCITY_THRESHOLD`` continuously for ``_SETTLE_WINDOW``
    and position within tolerance.

    This class runs **one** physics timestep level; the full dt sweep is
    orchestrated externally by the UI layer, which changes the physics dt between
    invocations and aggregates the per-level results via :func:`aggregate_dt_sweep`.

    Configurable via ``test_params``:
        timeout (float): Maximum sim-time seconds for the approach phase before
            declaring the joint did not settle. Default 10.0.
        hold_duration (float): Seconds to hold after settling (or after timeout if
            the joint never settled), used to measure steady-state error. Default 1.0.
        tolerance (float): Position error threshold (rad or m) for the settle check.
            Default 0.01.
        target_pct (float): Fraction of the half-range to offset the step command
            from the center of each joint's range. Default 0.25.
    """

    name = "Discretization Sweep"

    def __init__(self) -> None:
        super().__init__()
        self._timeout = 10.0
        self._hold_duration = 1.0
        self._tolerance = 0.01
        self._target_pct = 0.25
        self._joint_indices: list[int] = []
        self._joint_modes: dict[int, int] = {}
        self._test_params: dict = {}

    def setup(
        self, articulation: Articulation, joint_indices: list[int], joint_modes: dict[int, int], test_params: dict
    ) -> None:
        """Prepare the single-level discretization probe.

        Args:
            articulation: The Articulation wrapper for the robot under test.
            joint_indices: DOF indices being tested.
            joint_modes: Mapping of dof_index to JointMode value.
            test_params: Dict with optional keys listed in the class docstring.
        """
        super().setup(articulation, joint_indices, joint_modes, test_params)
        self._joint_indices = list(joint_indices)
        self._joint_modes = dict(joint_modes)
        self._test_params = test_params
        self._timeout = float(test_params.get("timeout", 10.0))
        self._hold_duration = float(test_params.get("hold_duration", 1.0))
        self._tolerance = float(test_params.get("tolerance", 0.01))
        self._target_pct = float(test_params.get("target_pct", 0.25))

    def _position_dof_indices(self) -> list[int]:
        """Collect the position-mode DOF indices commanded by this probe.

        Returns:
            Ordered, de-duplicated list of position-mode DOF indices drawn from
            the configured sequences (falling back to all tested joints).
        """
        pos_dof_idx: list[int] = []
        sequences = self._test_params.get("sequence", [{}])
        for seq in sequences:
            seq_joints = seq.get("joint_indices", np.array(self._joint_indices, dtype=np.int32))
            for idx in seq_joints:
                idx = int(idx)
                if self._joint_modes.get(idx, _JOINT_MODE_NONE) == _JOINT_MODE_POSITION and idx not in pos_dof_idx:
                    pos_dof_idx.append(idx)
        return pos_dof_idx

    def run(self) -> Generator[None, None, TestResult]:
        """Generator that executes the single-level accuracy probe.

        Commands the position-mode joints to a step target, waits for them to
        settle (or time out), then holds to measure the steady-state error.

        Yields:
            None, after each physics step.

        Returns:
            TestResult with recorded trajectories and per-joint settling / error metrics.
        """  # noqa: DOC405
        articulation = self._articulation
        if articulation is None:
            return TestResult(
                joint_position_commands=np.empty((0, 0)),
                joint_velocity_commands=np.empty((0, 0)),
                observed_joint_positions=np.empty((0, 0)),
                observed_joint_velocities=np.empty((0, 0)),
                command_times=np.empty(0),
            )

        pos_cmd_list: list[np.ndarray] = []
        vel_cmd_list: list[np.ndarray] = []
        obs_pos_list: list[np.ndarray] = []
        obs_vel_list: list[np.ndarray] = []
        time_list: list[float] = []

        lower_limits_all, upper_limits_all = [np.array(lim.list()) for lim in articulation.get_dof_limits()]

        pos_dof_idx = self._position_dof_indices()
        if not pos_dof_idx:
            articulation.reset_to_default_state()
            return TestResult(
                joint_position_commands=np.empty((0, 0)),
                joint_velocity_commands=np.empty((0, 0)),
                observed_joint_positions=np.empty((0, 0)),
                observed_joint_velocities=np.empty((0, 0)),
                command_times=np.empty(0),
            )

        lower = lower_limits_all[pos_dof_idx]
        upper = upper_limits_all[pos_dof_idx]
        center = (upper + lower) / 2.0
        half_range = (upper - lower) / 2.0
        step_target = center + self._target_pct * half_range
        np.clip(step_target, lower, upper, out=step_target)

        position_targets = articulation.get_dof_position_targets().numpy()[0].copy()
        velocity_targets = articulation.get_dof_velocity_targets().numpy()[0].copy()

        articulation.set_dof_position_targets(step_target, dof_indices=pos_dof_idx)
        position_targets[pos_dof_idx] = step_target

        low_vel_times: dict[int, float] = dict.fromkeys(pos_dof_idx, 0.0)
        joint_done: dict[int, bool] = dict.fromkeys(pos_dof_idx, False)
        joint_settle_time: dict[int, float] = {idx: float("nan") for idx in pos_dof_idx}
        hold_errors: dict[int, list[float]] = {idx: [] for idx in pos_dof_idx}

        total_time = 0.0

        pos_cmd_list.append(position_targets.copy())
        vel_cmd_list.append(velocity_targets.copy())
        time_list.append(total_time)
        obs_pos_list.append(articulation.get_dof_positions().numpy()[0].copy())
        obs_vel_list.append(articulation.get_dof_velocities().numpy()[0].copy())
        yield

        # Approach phase: wait for every joint to settle or for the approach to time out.
        in_approach = True
        while in_approach:
            dt = self.step
            total_time += dt

            positions = articulation.get_dof_positions().numpy()[0]
            velocities = articulation.get_dof_velocities().numpy()[0]

            for i, dof_idx in enumerate(pos_dof_idx):
                if joint_done[dof_idx]:
                    continue
                if abs(velocities[dof_idx]) <= _VELOCITY_THRESHOLD:
                    low_vel_times[dof_idx] += dt
                    if low_vel_times[dof_idx] >= _SETTLE_WINDOW:
                        joint_done[dof_idx] = True
                        if abs(positions[dof_idx] - step_target[i]) <= self._tolerance:
                            joint_settle_time[dof_idx] = total_time
                else:
                    low_vel_times[dof_idx] = 0.0

            all_done = all(joint_done[idx] for idx in pos_dof_idx)
            if all_done or total_time >= self._timeout:
                in_approach = False

            pos_cmd_list.append(position_targets.copy())
            vel_cmd_list.append(velocity_targets.copy())
            time_list.append(total_time)
            obs_pos_list.append(positions.copy())
            obs_vel_list.append(velocities.copy())

            if in_approach:
                yield

        yield

        # Hold phase: sample the steady-state position error.
        hold_start = total_time
        while total_time - hold_start < self._hold_duration:
            dt = self.step
            total_time += dt

            positions = articulation.get_dof_positions().numpy()[0]
            velocities = articulation.get_dof_velocities().numpy()[0]

            for i, dof_idx in enumerate(pos_dof_idx):
                hold_errors[dof_idx].append(abs(float(positions[dof_idx] - step_target[i])))

            pos_cmd_list.append(position_targets.copy())
            vel_cmd_list.append(velocity_targets.copy())
            time_list.append(total_time)
            obs_pos_list.append(positions.copy())
            obs_vel_list.append(velocities.copy())

            yield

        joint_metrics: dict[int, dict] = {}
        for dof_idx in pos_dof_idx:
            settle_t = joint_settle_time[dof_idx]
            settled = settle_t == settle_t  # not NaN
            errs = hold_errors[dof_idx]
            joint_metrics[dof_idx] = {
                "test_type": "discretization_sweep",
                "settling_time": float(settle_t) if settled else None,
                "steady_state_error": float(np.mean(errs)) if errs else float("nan"),
                "settled": settled,
            }

        articulation.reset_to_default_state()

        return TestResult(
            joint_position_commands=np.array(pos_cmd_list),
            joint_velocity_commands=np.array(vel_cmd_list),
            observed_joint_positions=np.array(obs_pos_list),
            observed_joint_velocities=np.array(obs_vel_list),
            command_times=np.array(time_list),
            joint_metrics=joint_metrics,
        )
