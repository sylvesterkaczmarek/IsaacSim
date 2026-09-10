# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Baseline and parameter-sensitivity diagnostics for automated SysID workflows."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .optimizer_base import OptimizerConfig
from .run_report import validation_summary
from .trajectory_resampling import combine_sample_weights
from .trajectory_segments import ChunkValidationMetric, TrajectoryChunk

ANALYSIS_SCHEMA_VERSION = 1

ProgressCallback = Callable[[str, float], None]


class _AnalysisRecommendation(str, Enum):
    """Stable recommendation codes emitted by analysis reports."""

    REVIEW_BASELINE_VALIDATION_GAP = "baseline_validation_gap_review_chunks_timing_or_model_before_broadening"
    CHECK_LOW_SENSITIVITY = "all_selected_parameters_low_sensitivity_check_mapping_articulation_or_residuals"
    REWEIGHT_OR_RECHUNK = "train_validation_sensitivity_disagrees_reweight_or_rechunk_before_tuning"
    REBALANCE_CHUNKS = "sensitivity_dominated_by_one_chunk_split_or_rebalance_chunks"
    RUN_SINGLE_PARAMETER_FAMILY = "run_next_conservative_solve_with_supported_single_parameter_family"
    NO_CLEAR_NEXT_STEP = "baseline_and_sensitivity_do_not_identify_a_clear_parameter_next_step"


@dataclass(frozen=True)
class ParameterPerturbation:
    """One-at-a-time finite-difference points for a selected parameter."""

    index: int
    name: str
    initial: float
    min: float
    max: float
    minus: float | None
    plus: float | None
    step: float
    scheme: str
    bound_limited: bool

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return asdict(self)


@dataclass
class AnalysisReplaySet:
    """Metrics and rollout data for one theta over a chunk set."""

    metrics: list[ChunkValidationMetric] = field(default_factory=list)
    rollouts: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            "metrics": [metric.to_dict() for metric in self.metrics],
            "summary": validation_summary(self.metrics),
        }


def build_parameter_perturbation(
    *,
    index: int,
    name: str,
    initial: float,
    min_value: float,
    max_value: float,
    relative_epsilon: float,
    absolute_epsilon: float,
) -> ParameterPerturbation:
    """Return finite-difference points clipped to parameter bounds.

    Args:
        index: Value supplied for ``index``.
        name: Value supplied for ``name``.
        initial: Value supplied for ``initial``.
        min_value: Value supplied for ``min_value``.
        max_value: Value supplied for ``max_value``.
        relative_epsilon: Value supplied for ``relative_epsilon``.
        absolute_epsilon: Value supplied for ``absolute_epsilon``.

    Returns:
        Result produced by the operation.
    """
    span = max(0.0, float(max_value) - float(min_value))
    raw_step = max(abs(float(initial)) * abs(float(relative_epsilon)), abs(float(absolute_epsilon)))
    step = min(raw_step, span) if span > 0.0 else raw_step
    lower = float(initial) - step
    upper = float(initial) + step
    minus = max(float(min_value), lower)
    plus = min(float(max_value), upper)
    minus_value = minus if minus < float(initial) else None
    plus_value = plus if plus > float(initial) else None
    bound_limited = minus_value is None or plus_value is None or minus != lower or plus != upper
    if minus_value is not None and plus_value is not None and not bound_limited:
        scheme = "centered"
    elif minus_value is not None or plus_value is not None:
        scheme = "one_sided"
    else:
        scheme = "unavailable"
    return ParameterPerturbation(
        index=int(index),
        name=name,
        initial=float(initial),
        min=float(min_value),
        max=float(max_value),
        minus=minus_value,
        plus=plus_value,
        step=float(step),
        scheme=scheme,
        bound_limited=bool(bound_limited),
    )


def perturb_theta(theta: list[float], perturbation: ParameterPerturbation, value: float) -> list[float]:  # noqa: D103
    updated = [float(item) for item in theta]
    updated[int(perturbation.index)] = float(value)
    return updated


async def evaluate_chunk_set(
    optimizer,  # noqa: ANN001
    config: OptimizerConfig,
    chunks: list[TrajectoryChunk],
    theta: list[float] | torch.Tensor,
    *,
    label: str,
    on_progress: ProgressCallback | None = None,
) -> AnalysisReplaySet:
    """Evaluate one theta on every chunk and collect metrics plus rollout plots.

    Args:
        optimizer: Optimizer used by the operation.
        config: Configuration for the operation.
        chunks: Trajectory chunks used by the operation.
        theta: Parameter values evaluated by the operation.
        label: Value supplied for ``label``.
        on_progress: Optional progress callback.

    Returns:
        Result produced by the operation.
    """  # noqa: DOC107
    result = AnalysisReplaySet()
    total = max(1, len(chunks))
    for index, chunk in enumerate(chunks):
        start = index / total
        end = (index + 1) / total

        def chunk_progress(message: str, fraction: float, *, _start=start, _end=end) -> None:  # noqa: ANN001
            if on_progress is None:
                return
            local = max(0.0, min(1.0, float(fraction)))
            on_progress(f"{label} {chunk.spec.display_name(index)}: {message}", _start + (_end - _start) * local)

        replay_fn = getattr(optimizer, "evaluate_validation_chunk_with_rollout", None)
        if callable(replay_fn):
            replay = await replay_fn(
                config,
                chunk,
                theta,
                on_progress=chunk_progress if on_progress is not None else None,
            )
            result.metrics.append(replay.metric)
            if replay.rollout is not None:
                result.rollouts.append(replay.rollout)
            continue
        metric = await optimizer.evaluate_validation_chunk(
            config,
            chunk,
            theta,
            on_progress=chunk_progress if on_progress is not None else None,
        )
        result.metrics.append(metric)
    return result


PARAMETER_CONFIDENCE_SCHEMA_VERSION = 2

CONFIDENCE_WELL_CONSTRAINED = "well_constrained"
CONFIDENCE_WEAKLY_CONSTRAINED = "weakly_constrained"
CONFIDENCE_BOUND_LIMITED = "bound_limited"
CONFIDENCE_UNKNOWN = "unknown"

_WELL_CONSTRAINED_SPAN_FRACTION = 0.1

#: Maximum parallel worlds per confidence cost batch; bounds the replicated
#: model's per-substep state memory (world-per-row) on small GPUs.
_CONFIDENCE_MAX_WORLDS = 32

NOISE_FLOOR_COST_FACTOR = 3.0
NOISE_FLOOR_METHOD = "residual_whiteness_second_difference"
NOISE_FLOOR_NOTE = "at noise floor — verdicts unreliable, validate against ground truth or load-side torque"

_NOISE_FLOOR_MIN_SAMPLES = 16


@dataclass
class NoiseFloorAssessment:
    """Validity gate comparing the solve cost against the estimated measurement-noise floor.

    Curvature-based confidence verdicts measure sensitivity of the residual to each
    parameter. When the residual is dominated by measurement noise (for example a
    feedforward supplies the motion torque and the fit reaches the sensor floor),
    that curvature reflects noise coupling rather than identifiability, so the
    per-parameter verdicts must not be trusted. The floor is estimated from the
    high-frequency power of the nominal-rollout residual (second differences with a
    robust MAD scale): a white, noise-dominated residual has total power within a
    small factor of its high-frequency power, while structured model error is
    concentrated at low frequencies. Low-pass-filtered telemetry noise can evade
    this estimator, so a clear ``at_noise_floor=False`` is necessary but not
    sufficient evidence that the verdicts are reliable.
    """

    estimated: bool = False
    method: str = NOISE_FLOOR_METHOD
    baseline_cost: float | None = None
    noise_floor_cost: float | None = None
    cost_to_floor_ratio: float | None = None
    factor_threshold: float = NOISE_FLOOR_COST_FACTOR
    at_noise_floor: bool = False
    channels: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return asdict(self)


def estimate_series_noise_power(series: Any) -> float | None:
    """Estimate white-noise power of a time series from its high-frequency content.

    Uses second differences (which annihilate slowly varying structure) with a
    Gaussian-consistent MAD scale so occasional structured transients do not
    inflate the estimate. For white noise of variance ``s^2`` the second
    difference has variance ``6 s^2``.

    Returns:
        Estimated noise variance, or None when the series is too short.

    Args:
        series: Value supplied for ``series``.
    """
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if values.shape[0] < _NOISE_FLOOR_MIN_SAMPLES:
        return None
    second_diff = values[2:] - 2.0 * values[1:-1] + values[:-2]
    median = float(np.median(second_diff))
    mad = float(np.median(np.abs(second_diff - median)))
    sigma = 1.4826 * mad
    return sigma * sigma / 6.0


def assess_noise_floor(
    nominal_plot: Any,
    residual_weight_config: Any,
    baseline_cost: float,
    *,
    factor_threshold: float = NOISE_FLOOR_COST_FACTOR,
    sample_weights: np.ndarray | None = None,
) -> NoiseFloorAssessment:
    """Compare the nominal-rollout residual power against its estimated noise floor.

    Args:
        nominal_plot: ``SysIdRolloutPlotData`` for the baseline theta (env 0) with
            joint-major measured/simulated series, or None when unavailable.
        residual_weight_config: ``ResidualWeightConfig`` (or None for the default
            position/velocity weights) used to weight channels like the cost does.
        baseline_cost: Scalar cost at the identified theta, used only to express
            the floor in cost units.
        factor_threshold: Residual power within this factor of the estimated
            high-frequency floor flags the solve as at the noise floor.
        sample_weights: Optional residual multipliers applied to each sample.

    Returns:
        Assessment with the aggregate ratio, per-channel detail, and the
        ``at_noise_floor`` flag. ``estimated`` is False when no usable series
        were available (missing plot data or too few samples).
    """
    return assess_segmented_noise_floor(
        [(nominal_plot, sample_weights, 1.0, "")],
        residual_weight_config,
        baseline_cost,
        factor_threshold=factor_threshold,
    )


def assess_segmented_noise_floor(
    segments: list[tuple[Any, np.ndarray | None, float, str]],
    residual_weight_config: Any,
    baseline_cost: float,
    *,
    factor_threshold: float = NOISE_FLOOR_COST_FACTOR,
) -> NoiseFloorAssessment:
    """Aggregate noise-floor evidence from weighted trajectory segments.

    Args:
        segments: Plot data, sample weights, segment weight, and label for each segment.
        residual_weight_config: Residual-channel weights used for the solve.
        baseline_cost: Optimized segmented cost.
        factor_threshold: Largest cost-to-noise-floor ratio considered noise dominated.

    Returns:
        Aggregate noise-floor assessment and per-channel evidence.
    """
    assessment = NoiseFloorAssessment(
        baseline_cost=float(baseline_cost) if _finite(baseline_cost) else None,
        factor_threshold=float(factor_threshold),
    )
    available = [(plot, weights, max(float(weight), 0.0), label) for plot, weights, weight, label in segments if plot]
    total_segment_weight = sum(weight for _plot, _weights, weight, _label in available)
    if not available or total_segment_weight <= 0.0:
        assessment.note = "noise floor not estimated: no nominal rollout plot data"
        return assessment

    total_power = 0.0
    total_floor = 0.0
    for nominal_plot, sample_weights, segment_weight, label in available:
        normalized_segment_weight = segment_weight / total_segment_weight
        rows, segment_power, segment_floor = _noise_floor_components(
            nominal_plot,
            residual_weight_config,
            sample_weights=sample_weights,
            source=label,
        )
        assessment.channels.extend(rows)
        total_power += normalized_segment_weight * segment_power
        total_floor += normalized_segment_weight * segment_floor

    if not assessment.channels:
        assessment.note = "noise floor not estimated: no usable residual series"
        return assessment

    assessment.estimated = True
    if total_floor <= 0.0:
        # Exactly zero high-frequency content (e.g. noiseless sim-sim data): any
        # remaining residual is structure, so the verdicts stand.
        return assessment
    ratio = total_power / total_floor
    assessment.cost_to_floor_ratio = ratio
    if assessment.baseline_cost is not None and ratio > 0.0:
        assessment.noise_floor_cost = assessment.baseline_cost / ratio
    if ratio <= factor_threshold:
        assessment.at_noise_floor = True
        assessment.note = NOISE_FLOOR_NOTE
    return assessment


def _noise_floor_components(
    nominal_plot: Any,
    residual_weight_config: Any,
    *,
    sample_weights: np.ndarray | None,
    source: str,
) -> tuple[list[dict[str, Any]], float, float]:
    position_weight = float(getattr(residual_weight_config, "position_weight", 1.0))
    velocity_weight = float(getattr(residual_weight_config, "velocity_weight", 1.0))
    torque_weight = float(getattr(residual_weight_config, "torque_weight", 0.0))
    signal_series = (
        ("position", position_weight, nominal_plot.measured_positions, nominal_plot.simulated_positions),
        ("velocity", velocity_weight, nominal_plot.measured_velocities, nominal_plot.simulated_velocities),
        ("torque", torque_weight, nominal_plot.measured_efforts, nominal_plot.simulated_efforts),
    )

    rows: list[dict[str, Any]] = []
    total_power = 0.0
    total_floor = 0.0
    for signal, weight, measured, simulated in signal_series:
        if weight <= 0.0 or not measured or not simulated:
            continue
        for joint_index, (meas, sim) in enumerate(zip(measured, simulated)):
            length = min(len(meas), len(sim))
            residual = np.asarray(sim[:length], dtype=np.float64) - np.asarray(meas[:length], dtype=np.float64)
            floor_power = estimate_series_noise_power(residual)
            if floor_power is None:
                continue
            residual_power = float(np.mean(residual * residual))
            if sample_weights is not None:
                applied = np.asarray(sample_weights[:length], dtype=np.float64).reshape(-1)
                if applied.shape[0] < length:
                    applied = np.pad(applied, (0, length - applied.shape[0]), mode="edge")
                applied = applied[:length]
                residual_power = float(np.mean((residual * applied) ** 2))
                floor_power *= float(np.mean(applied * applied))
            total_power += weight * residual_power
            total_floor += weight * floor_power
            rows.append(
                {
                    "source": source,
                    "signal": signal,
                    "joint": joint_index,
                    "weight": weight,
                    "sample_weighted": sample_weights is not None,
                    "residual_power": residual_power,
                    "noise_floor_power": floor_power,
                }
            )
    return rows, total_power, total_floor


@dataclass
class ParameterConfidenceEntry:
    """Post-solve sensitivity of the cost to one identified parameter.

    ``ten_percent_range`` is the parameter offset at which the (quadratic model of
    the) validation cost rises by 10% — a sensitivity range, not a statistical
    confidence interval.
    """

    index: int
    name: str
    param_type: str
    value: float
    step: float
    cost_baseline: float
    cost_minus: float | None
    cost_plus: float | None
    curvature: float | None
    ten_percent_range: float | None
    verdict: str
    bound_limited: bool
    #: True when the solve cost sits at the estimated measurement-noise floor, in
    #: which case ``verdict`` reflects noise coupling and must not be trusted.
    at_noise_floor: bool = False

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return asdict(self)


@dataclass
class ParameterConfidenceReport:
    """Per-parameter post-solve sensitivity over a chunk set."""

    schema_version: int = PARAMETER_CONFIDENCE_SCHEMA_VERSION
    entries: list[ParameterConfidenceEntry] = field(default_factory=list)
    chunk_source: str = ""
    rollout_samples_used: int = 0
    budget_capped: bool = False
    skipped_reason: str = ""
    noise_floor: NoiseFloorAssessment = field(default_factory=NoiseFloorAssessment)

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            "schema_version": self.schema_version,
            "entries": [entry.to_dict() for entry in self.entries],
            "chunk_source": self.chunk_source,
            "rollout_samples_used": self.rollout_samples_used,
            "budget_capped": self.budget_capped,
            "skipped_reason": self.skipped_reason,
            "noise_floor": self.noise_floor.to_dict(),
        }

    def summary_lines(self) -> list[str]:  # noqa: D102
        lines: list[str] = []
        for entry in self.entries:
            rng = f"; sens. span {entry.ten_percent_range:.4g}" if entry.ten_percent_range is not None else ""
            marker = "; at noise floor" if entry.at_noise_floor else ""
            lines.append(f"{entry.name}: {entry.value:.5g} ({entry.verdict}{rng}{marker})")
        if self.noise_floor.at_noise_floor:
            ratio = self.noise_floor.cost_to_floor_ratio
            ratio_text = f" (cost/floor ratio {ratio:.2g})" if ratio is not None else ""
            lines.append(f"Solve cost is at the estimated measurement-noise floor{ratio_text}: {NOISE_FLOOR_NOTE}.")
        if self.skipped_reason:
            lines.append(f"Confidence estimation skipped: {self.skipped_reason}")
        return lines


async def build_parameter_confidence_report(
    optimizer,  # noqa: ANN001
    config: OptimizerConfig,
    chunks: list[TrajectoryChunk],
    theta_hat: list[float],
    *,
    relative_epsilon: float = 0.05,
    absolute_epsilon: float = 1e-4,
    max_rollout_sample_budget: int = 2_000_000,
    on_progress: ProgressCallback | None = None,
) -> ParameterConfidenceReport:
    """Estimate per-parameter sensitivity ranges around the identified theta.

    Perturbs each parameter of ``theta_hat`` by a bound-clipped step, evaluates the
    chunk-set cost for every perturbed vector in one batched pass through the
    optimizer's cost function, and reports the curvature-based range where the cost
    rises 10%. Rollout work is capped by ``max_rollout_sample_budget`` (samples =
    steps x evaluations): over budget the minus side is dropped (one-sided), and if
    still over the estimation is skipped with a reason.

    The verdicts are gated by a noise-floor check (see ``NoiseFloorAssessment``):
    when the baseline residual is indistinguishable from measurement noise, every
    entry is marked ``at_noise_floor`` and the report carries an explicit warning,
    because curvature then measures noise coupling instead of identifiability.

    Args:
        optimizer: Optimizer used by the operation.
        config: Configuration for the operation.
        chunks: Trajectory chunks used by the operation.
        theta_hat: Value supplied for ``theta_hat``.
        relative_epsilon: Value supplied for ``relative_epsilon``.
        absolute_epsilon: Value supplied for ``absolute_epsilon``.
        max_rollout_sample_budget: Value supplied for ``max_rollout_sample_budget``.
        on_progress: Optional progress callback.

    Returns:
        Result produced by the operation.
    """  # noqa: DOC107
    from dataclasses import replace as dataclass_replace

    report = ParameterConfidenceReport()
    if not config.param_entries or not chunks:
        report.skipped_reason = "no parameters or chunks to evaluate"
        return report

    theta_values = [float(value) for value in theta_hat]
    perturbations = [
        build_parameter_perturbation(
            index=row["index"],
            name=row["name"],
            initial=theta_values[row["index"]],
            min_value=row["min"],
            max_value=row["max"],
            relative_epsilon=relative_epsilon,
            absolute_epsilon=absolute_epsilon,
        )
        for row in parameter_rows(config)
    ]

    samples_per_eval = sum(min(int(chunk.sample_count), int(config.max_rollout_steps)) for chunk in chunks)
    samples_per_eval = max(1, samples_per_eval)
    extra_nominal_samples = sum(min(int(chunk.sample_count), int(config.max_rollout_steps)) for chunk in chunks[1:])

    def planned_evaluations(use_minus: bool) -> int:
        count = 1  # baseline
        for perturbation in perturbations:
            if perturbation.plus is not None:
                count += 1
            if use_minus and perturbation.minus is not None:
                count += 1
        return count

    def planned_samples(use_minus: bool) -> int:
        return planned_evaluations(use_minus) * samples_per_eval + extra_nominal_samples

    use_minus = True
    if planned_samples(True) > max_rollout_sample_budget:
        use_minus = False
        report.budget_capped = True
        if planned_samples(False) > max_rollout_sample_budget:
            report.skipped_reason = "rollout sample budget exceeded"
            return report

    rows: list[list[float]] = [list(theta_values)]
    row_of: dict[tuple[int, str], int] = {}
    for perturbation in perturbations:
        if perturbation.plus is not None:
            row_of[(perturbation.index, "plus")] = len(rows)
            rows.append(perturb_theta(theta_values, perturbation, perturbation.plus))
        if use_minus and perturbation.minus is not None:
            row_of[(perturbation.index, "minus")] = len(rows)
            rows.append(perturb_theta(theta_values, perturbation, perturbation.minus))

    device = getattr(getattr(optimizer, "_bridge", None), "device", torch.device("cpu"))
    theta_batch = torch.tensor(rows, dtype=torch.float32, device=device)
    conf_config = dataclass_replace(
        config,
        trajectory=chunks[0].trajectory,
        training_segments=list(chunks),
        training_chunks=None,
        training_chunk_weights=None,
        training_chunk_labels=None,
    )
    if on_progress is not None:
        on_progress("Estimating per-parameter confidence...", 0.0)
    # Plot data is requested for the nominal rollout (env 0 = theta_hat): its
    # measured/simulated series feed the noise-floor validity gate below.
    # The batch is evaluated in bounded world groups: rolling baseline plus all
    # perturbations as one world-per-row batch allocates per-substep state
    # buffers for every world simultaneously (~7 GiB for 65 parameters at
    # 800x32 on a 7-DOF arm) and OOMs 16 GB cards after a solve. Group costs
    # are identical to the single-batch result — worlds are independent.
    group_size = _CONFIDENCE_MAX_WORLDS
    costs_list: list[float] = []
    nominal_plot = None
    for group_start in range(0, len(rows), group_size):
        group = theta_batch[group_start : group_start + group_size]
        pad_rows = 0
        if group_start > 0 and group.shape[0] < group_size:
            # Reuse the first group's world-count context instead of building a
            # second replicated model for the remainder.
            pad_rows = group_size - int(group.shape[0])
            group = torch.cat([group, group[-1:].expand(pad_rows, -1)], dim=0)
        group_costs, _residuals, group_plot = await optimizer.compute_costs_for_theta_batch(
            conf_config,
            group,
            on_progress=on_progress if group_start == 0 else None,
            include_plot_data=(group_start == 0),
        )
        if group_start == 0:
            nominal_plot = group_plot
        group_values = [float(value) for value in group_costs.detach().cpu().reshape(-1).tolist()]
        if pad_rows:
            group_values = group_values[: len(group_values) - pad_rows]
        costs_list.extend(group_values)
    nominal_segments: list[tuple[Any, np.ndarray | None, float, str]] = []
    first_steps = min(int(chunks[0].sample_count), int(config.max_rollout_steps))
    first_weights = combine_sample_weights(
        getattr(config.residual_weight_config, "sample_weights", None),
        getattr(chunks[0].trajectory, "residual_sample_weights", None),
        num_steps=first_steps,
    )
    nominal_segments.append((nominal_plot, first_weights, float(chunks[0].spec.weight), chunks[0].spec.display_name(0)))

    # Segmented training returns plot data for the first segment only. Replay the
    # nominal theta once for each remaining chunk so the global noise-floor gate
    # cannot depend on chunk ordering.
    for chunk_index, chunk in enumerate(chunks[1:], start=1):
        chunk_config = dataclass_replace(
            config,
            trajectory=chunk.trajectory,
            training_segments=None,
            training_chunks=None,
            training_chunk_weights=None,
            training_chunk_labels=None,
        )
        _chunk_cost, _chunk_residuals, chunk_plot = await optimizer.compute_costs_for_theta_batch(
            chunk_config,
            theta_batch[:1],
            on_progress=None,
            include_plot_data=True,
        )
        chunk_steps = min(int(chunk.sample_count), int(config.max_rollout_steps))
        chunk_weights = combine_sample_weights(
            getattr(config.residual_weight_config, "sample_weights", None),
            getattr(chunk.trajectory, "residual_sample_weights", None),
            num_steps=chunk_steps,
        )
        nominal_segments.append(
            (chunk_plot, chunk_weights, float(chunk.spec.weight), chunk.spec.display_name(chunk_index))
        )

    report.rollout_samples_used = len(rows) * samples_per_eval + extra_nominal_samples

    baseline_cost = costs_list[0]
    report.noise_floor = assess_segmented_noise_floor(
        nominal_segments,
        config.residual_weight_config,
        baseline_cost,
    )
    for index, perturbation in enumerate(perturbations):
        cost_plus = _cost_at(costs_list, row_of, perturbation.index, "plus")
        cost_minus = _cost_at(costs_list, row_of, perturbation.index, "minus")
        curvature = _quadratic_curvature(perturbation, baseline_cost, cost_minus, cost_plus)
        ten_percent_range = None
        if curvature is not None and curvature > 0.0 and math.isfinite(curvature) and baseline_cost > 0.0:
            ten_percent_range = math.sqrt(0.2 * baseline_cost / curvature)
        entry = ParameterConfidenceEntry(
            index=perturbation.index,
            name=perturbation.name,
            param_type=(
                config.param_entries[index].param_type.value
                if hasattr(config.param_entries[index].param_type, "value")
                else str(config.param_entries[index].param_type)
            ),
            value=theta_values[perturbation.index],
            step=perturbation.step,
            cost_baseline=baseline_cost,
            cost_minus=cost_minus,
            cost_plus=cost_plus,
            curvature=curvature,
            ten_percent_range=ten_percent_range,
            verdict=_confidence_verdict(perturbation, curvature, ten_percent_range),
            bound_limited=perturbation.bound_limited,
            at_noise_floor=report.noise_floor.at_noise_floor,
        )
        report.entries.append(entry)
    if on_progress is not None:
        on_progress("Per-parameter confidence complete", 1.0)
    return report


def _cost_at(costs: list[float], row_of: dict[tuple[int, str], int], index: int, side: str) -> float | None:
    row = row_of.get((index, side))
    if row is None or row >= len(costs):
        return None
    value = costs[row]
    return float(value) if math.isfinite(value) else None


def _quadratic_curvature(
    perturbation: ParameterPerturbation,
    baseline: float,
    cost_minus: float | None,
    cost_plus: float | None,
) -> float | None:
    if not math.isfinite(baseline):
        return None
    step = float(perturbation.step)
    if step <= 0.0:
        return None
    if cost_plus is not None and cost_minus is not None and perturbation.scheme == "centered":
        return (cost_plus + cost_minus - 2.0 * baseline) / (step * step)
    one_sided = cost_plus if cost_plus is not None else cost_minus
    if one_sided is None:
        return None
    point = perturbation.plus if cost_plus is not None else perturbation.minus
    offset = abs(float(point) - perturbation.initial)
    if offset <= 0.0:
        return None
    # One-sided quadratic estimate assuming a stationary point at theta_hat.
    return 2.0 * (one_sided - baseline) / (offset * offset)


def _confidence_verdict(
    perturbation: ParameterPerturbation,
    curvature: float | None,
    ten_percent_range: float | None,
) -> str:
    if curvature is None or not math.isfinite(curvature) or curvature <= 0.0 or ten_percent_range is None:
        if perturbation.scheme == "unavailable" or perturbation.bound_limited:
            return CONFIDENCE_BOUND_LIMITED
        return CONFIDENCE_UNKNOWN
    if perturbation.bound_limited and perturbation.scheme != "centered":
        return CONFIDENCE_BOUND_LIMITED
    span = max(perturbation.max - perturbation.min, 1e-12)
    if ten_percent_range <= _WELL_CONSTRAINED_SPAN_FRACTION * span:
        return CONFIDENCE_WELL_CONSTRAINED
    return CONFIDENCE_WEAKLY_CONSTRAINED


def parameter_rows(config: OptimizerConfig) -> list[dict[str, Any]]:  # noqa: D103
    rows: list[dict[str, Any]] = []
    initials = _tensor_list(config.theta_initial)
    mins = _tensor_list(config.theta_min)
    maxs = _tensor_list(config.theta_max)
    for index, entry in enumerate(config.param_entries):
        name = entry.display_name(config.trajectory.num_joints) if hasattr(entry, "display_name") else str(entry)
        rows.append(
            {
                "index": int(index),
                "name": name,
                "param_type": entry.param_type.value if hasattr(entry.param_type, "value") else str(entry.param_type),
                "initial": float(initials[index]),
                "min": float(mins[index]),
                "max": float(maxs[index]),
            }
        )
    return rows


def build_perturbations(  # noqa: D103
    config: OptimizerConfig,
    *,
    relative_epsilon: float,
    absolute_epsilon: float,
) -> list[ParameterPerturbation]:
    rows = parameter_rows(config)
    return [
        build_parameter_perturbation(
            index=row["index"],
            name=row["name"],
            initial=row["initial"],
            min_value=row["min"],
            max_value=row["max"],
            relative_epsilon=relative_epsilon,
            absolute_epsilon=absolute_epsilon,
        )
        for row in rows
    ]


def metric_group_summary(metrics: list[ChunkValidationMetric]) -> dict[str, float | None]:  # noqa: D103
    return validation_summary(metrics).get("overall", {})


def build_sensitivity_entry(
    perturbation: ParameterPerturbation,
    *,
    baseline_train: list[ChunkValidationMetric],
    baseline_validation: list[ChunkValidationMetric],
    minus_train: list[ChunkValidationMetric] | None,
    minus_validation: list[ChunkValidationMetric] | None,
    plus_train: list[ChunkValidationMetric] | None,
    plus_validation: list[ChunkValidationMetric] | None,
) -> dict[str, Any]:
    """Create report rows for one parameter perturbation.

    Args:
        perturbation: Value supplied for ``perturbation``.
        baseline_train: Value supplied for ``baseline_train``.
        baseline_validation: Value supplied for ``baseline_validation``.
        minus_train: Value supplied for ``minus_train``.
        minus_validation: Value supplied for ``minus_validation``.
        plus_train: Value supplied for ``plus_train``.
        plus_validation: Value supplied for ``plus_validation``.

    Returns:
        Result produced by the operation.
    """
    train_gradient = _group_gradient(perturbation, baseline_train, minus_train, plus_train)
    validation_gradient = _group_gradient(perturbation, baseline_validation, minus_validation, plus_validation)
    train_minus_delta = _group_delta(baseline_train, minus_train)
    train_plus_delta = _group_delta(baseline_train, plus_train)
    validation_minus_delta = _group_delta(baseline_validation, minus_validation)
    validation_plus_delta = _group_delta(baseline_validation, plus_validation)
    chunk_rows = []
    baseline_by_key = _metrics_by_key(baseline_train + baseline_validation)
    minus_by_key = _metrics_by_key((minus_train or []) + (minus_validation or []))
    plus_by_key = _metrics_by_key((plus_train or []) + (plus_validation or []))
    for key, base_metric in baseline_by_key.items():
        chunk_rows.append(
            _chunk_sensitivity_row(
                perturbation,
                base_metric,
                minus_by_key.get(key),
                plus_by_key.get(key),
            )
        )
    dominance = _chunk_dominance(chunk_rows)
    sign_agreement = _sign_agreement(train_gradient.get("normalized_cost"), validation_gradient.get("normalized_cost"))
    low_sensitivity = _low_sensitivity(train_gradient, validation_gradient)
    return {
        "parameter": perturbation.to_dict(),
        "train_gradient": train_gradient,
        "validation_gradient": validation_gradient,
        "train_minus_delta": train_minus_delta,
        "train_plus_delta": train_plus_delta,
        "validation_minus_delta": validation_minus_delta,
        "validation_plus_delta": validation_plus_delta,
        "train_validation_gradient_agreement": sign_agreement,
        "chunk_sensitivities": chunk_rows,
        "dominant_chunk": dominance,
        "low_sensitivity": low_sensitivity,
        "bound_limited": perturbation.bound_limited,
    }


def build_analysis_recommendations(  # noqa: D103
    *,
    baseline: dict[str, Any] | None,
    sensitivity: list[dict[str, Any]],
) -> list[str]:
    recommendations: list[str] = []
    baseline_validation = _nested_metric(baseline, "validation", "summary", "overall", "normalized_cost")
    baseline_train = _nested_metric(baseline, "train", "summary", "overall", "normalized_cost")
    if _finite(baseline_validation) and _finite(baseline_train):
        ratio = float(baseline_validation) / max(float(baseline_train), 1e-12)
        if ratio > 10.0:
            recommendations.append(_AnalysisRecommendation.REVIEW_BASELINE_VALIDATION_GAP.value)
    if sensitivity:
        if all(bool(row.get("low_sensitivity")) for row in sensitivity):
            recommendations.append(_AnalysisRecommendation.CHECK_LOW_SENSITIVITY.value)
        if any(row.get("train_validation_gradient_agreement") == "opposite_sign" for row in sensitivity):
            recommendations.append(_AnalysisRecommendation.REWEIGHT_OR_RECHUNK.value)
        if any(row.get("dominant_chunk", {}).get("dominance_ratio", 0.0) > 0.7 for row in sensitivity):
            recommendations.append(_AnalysisRecommendation.REBALANCE_CHUNKS.value)
        families = sorted(
            {row.get("parameter", {}).get("name", "") for row in sensitivity if not row.get("low_sensitivity")}
        )
        if families and not any("disagrees" in item for item in recommendations):
            recommendations.append(_AnalysisRecommendation.RUN_SINGLE_PARAMETER_FAMILY.value)
    if not recommendations:
        recommendations.append(_AnalysisRecommendation.NO_CLEAR_NEXT_STEP.value)
    return recommendations


def _group_gradient(
    perturbation: ParameterPerturbation,
    baseline: list[ChunkValidationMetric],
    minus: list[ChunkValidationMetric] | None,
    plus: list[ChunkValidationMetric] | None,
) -> dict[str, float | None]:
    base_summary = metric_group_summary(baseline)
    minus_summary = metric_group_summary(minus or [])
    plus_summary = metric_group_summary(plus or [])
    return {
        metric_name: _finite_difference(
            perturbation,
            base_summary.get(metric_name),
            minus_summary.get(metric_name),
            plus_summary.get(metric_name),
        )
        for metric_name in (
            "normalized_cost",
            "position_rmse",
            "velocity_rmse",
            "torque_rmse",
            "end_effector_pose_rmse",
            "contact_force_rmse",
        )
    }


def _group_delta(
    baseline: list[ChunkValidationMetric],
    comparison: list[ChunkValidationMetric] | None,
) -> dict[str, float | None]:
    base_summary = metric_group_summary(baseline)
    comparison_summary = metric_group_summary(comparison or [])
    return {
        metric_name: _metric_delta(base_summary.get(metric_name), comparison_summary.get(metric_name))
        for metric_name in (
            "normalized_cost",
            "position_rmse",
            "velocity_rmse",
            "torque_rmse",
            "end_effector_pose_rmse",
            "contact_force_rmse",
        )
    }


def _chunk_sensitivity_row(
    perturbation: ParameterPerturbation,
    baseline: ChunkValidationMetric,
    minus: ChunkValidationMetric | None,
    plus: ChunkValidationMetric | None,
) -> dict[str, Any]:
    return {
        "name": baseline.name,
        "role": baseline.role,
        "excitation": baseline.excitation,
        "minus_delta": _metric_delta_row(baseline, minus),
        "plus_delta": _metric_delta_row(baseline, plus),
        "gradient": {
            metric_name: _finite_difference(
                perturbation,
                getattr(baseline, metric_name, None),
                getattr(minus, metric_name, None) if minus is not None else None,
                getattr(plus, metric_name, None) if plus is not None else None,
            )
            for metric_name in (
                "normalized_cost",
                "position_rmse",
                "velocity_rmse",
                "torque_rmse",
                "end_effector_pose_rmse",
                "contact_force_rmse",
            )
        },
    }


def _metric_delta_row(
    baseline: ChunkValidationMetric,
    comparison: ChunkValidationMetric | None,
) -> dict[str, float | None]:
    return {
        metric_name: _metric_delta(
            getattr(baseline, metric_name, None),
            getattr(comparison, metric_name, None) if comparison is not None else None,
        )
        for metric_name in (
            "normalized_cost",
            "position_rmse",
            "velocity_rmse",
            "torque_rmse",
            "end_effector_pose_rmse",
            "contact_force_rmse",
        )
    }


def _metric_delta(base_value: float | None, comparison_value: float | None) -> float | None:
    if not _finite(base_value) or not _finite(comparison_value):
        return None
    return float(comparison_value) - float(base_value)


def _finite_difference(
    perturbation: ParameterPerturbation,
    base_value: float | None,
    minus_value: float | None,
    plus_value: float | None,
) -> float | None:
    if (
        perturbation.minus is not None
        and perturbation.plus is not None
        and _finite(minus_value)
        and _finite(plus_value)
    ):
        denom = float(perturbation.plus) - float(perturbation.minus)
        return (float(plus_value) - float(minus_value)) / denom if abs(denom) > 0.0 else None
    if perturbation.plus is not None and _finite(base_value) and _finite(plus_value):
        denom = float(perturbation.plus) - float(perturbation.initial)
        return (float(plus_value) - float(base_value)) / denom if abs(denom) > 0.0 else None
    if perturbation.minus is not None and _finite(base_value) and _finite(minus_value):
        denom = float(perturbation.initial) - float(perturbation.minus)
        return (float(base_value) - float(minus_value)) / denom if abs(denom) > 0.0 else None
    return None


def _metrics_by_key(
    metrics: list[ChunkValidationMetric],
) -> dict[tuple[str, str, str, float, float], ChunkValidationMetric]:
    return {
        (
            metric.role,
            metric.name,
            metric.excitation,
            float(metric.start),
            float(metric.end),
        ): metric
        for metric in metrics
    }


def _chunk_dominance(chunk_rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for row in chunk_rows:
        gradient = row.get("gradient", {}).get("normalized_cost")
        if _finite(gradient):
            values.append((abs(float(gradient)), row))
    total = sum(value for value, _row in values)
    if total <= 0.0 or not values:
        return {"name": "", "role": "", "dominance_ratio": 0.0}
    value, row = max(values, key=lambda item: item[0])
    return {
        "name": row.get("name", ""),
        "role": row.get("role", ""),
        "dominance_ratio": float(value / total),
    }


def _sign_agreement(train_gradient: float | None, validation_gradient: float | None) -> str:
    if not _finite(train_gradient) or not _finite(validation_gradient):
        return "unavailable"
    train = float(train_gradient)
    validation = float(validation_gradient)
    if abs(train) < 1e-12 or abs(validation) < 1e-12:
        return "weak"
    return "same_sign" if train * validation > 0.0 else "opposite_sign"


def _low_sensitivity(train_gradient: dict[str, float | None], validation_gradient: dict[str, float | None]) -> bool:
    values = [
        abs(float(value))
        for value in (train_gradient.get("normalized_cost"), validation_gradient.get("normalized_cost"))
        if _finite(value)
    ]
    return not values or max(values) < 1e-6


def _nested_metric(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload or {}
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _tensor_list(value) -> list[float]:  # noqa: ANN001
    if hasattr(value, "detach"):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [float(item) for item in value]


def _finite(value) -> bool:  # noqa: ANN001
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
