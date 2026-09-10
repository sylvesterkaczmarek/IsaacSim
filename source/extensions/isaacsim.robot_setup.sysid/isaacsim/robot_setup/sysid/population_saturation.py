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

"""Population-saturation benchmarking for batched SysID rollouts.

The benchmark deliberately keeps candidate zero fixed while varying every other
candidate between replays.  Candidate zero is therefore both a useful latency
baseline and a sentinel for accidental cross-world coupling.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

DEFAULT_POPULATIONS = (1, 2, 4, 8, 12, 13, 16, 24, 32, 48, 64)


@dataclass(frozen=True)
class PopulationSaturationConfig:
    """Controls one population-size sweep."""

    populations: tuple[int, ...] = DEFAULT_POPULATIONS
    steady_replays: int = 20
    capture_expected: bool = True
    release_contexts: bool = True
    max_device_memory_fraction: float = 0.90
    minimum_free_device_bytes: int = 512 * 1024 * 1024
    saturation_efficiency_percent: float = 10.0

    def normalized_populations(self) -> tuple[int, ...]:
        """Return the validated, sorted population sizes.

        Returns:
            Unique positive population sizes in ascending order.
        """
        populations = tuple(sorted({int(value) for value in self.populations if int(value) > 0}))
        if not populations:
            raise ValueError("At least one positive population size is required.")
        if int(self.steady_replays) < 1:
            raise ValueError("steady_replays must be at least one.")
        if not 0.0 < float(self.max_device_memory_fraction) <= 1.0:
            raise ValueError("max_device_memory_fraction must be in (0, 1].")
        if not 0.0 <= float(self.saturation_efficiency_percent) <= 100.0:
            raise ValueError("saturation_efficiency_percent must be in [0, 100].")
        return populations


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    """Device-wide CUDA memory counters, including non-Torch allocations."""

    free_bytes: int = 0
    total_bytes: int = 0

    @property
    def used_bytes(self) -> int:
        """Return the number of allocated device bytes."""
        return max(0, int(self.total_bytes) - int(self.free_bytes))

    @property
    def used_fraction(self) -> float:
        """Return the fraction of total device memory in use."""
        return self.used_bytes / self.total_bytes if self.total_bytes > 0 else 0.0


@dataclass
class PopulationSaturationSample:
    """Measurements collected for one candidate population size."""

    population: int
    status: str
    warmup_s: float | None = None
    capture_build_s: float | None = None
    replay_samples_s: list[float] = field(default_factory=list)
    median_s: float | None = None
    p95_s: float | None = None
    candidates_per_s: float | None = None
    wall_s_per_candidate: float | None = None
    aggregate_rt: float | None = None
    parallel_efficiency: float | None = None
    marginal_throughput_gain_percent: float | None = None
    marginal_scaling_efficiency_percent: float | None = None
    finite: bool = False
    graph_count: int | None = None
    capture_failed: bool | None = None
    candidate0_position_rmse: float | None = None
    candidate0_velocity_rmse: float | None = None
    candidate0_torque_rmse: float | None = None
    device_used_before_bytes: int = 0
    device_used_after_capture_bytes: int = 0
    device_used_after_replays_bytes: int = 0
    device_used_after_release_bytes: int = 0
    replay_memory_drift_bytes: int = 0
    error: str = ""


@dataclass
class PopulationSaturationReport:
    """Results of a complete population-saturation sweep."""

    duration_s_per_candidate: float
    num_steps: int
    num_parameters: int
    steady_replays: int
    capture_expected: bool
    samples: list[PopulationSaturationSample] = field(default_factory=list)
    saturation_population: int | None = None
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation.

        Returns:
            The recursively serialized report.
        """
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report as JSON.

        Args:
            indent: JSON indentation width.

        Returns:
            The serialized report.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path) -> Path:
        """Write the report to a file.

        Args:
            path: Destination JSON path.

        Returns:
            The normalized output path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output


def build_diverse_candidate_population(
    theta_initial: torch.Tensor,
    theta_min: torch.Tensor,
    theta_max: torch.Tensor,
    population: int,
    *,
    generation: int = 0,
) -> torch.Tensor:
    """Build a deterministic, bounded population whose first row is invariant.

    Args:
        theta_initial: Initial parameter vector retained as candidate zero.
        theta_min: Lower parameter bounds.
        theta_max: Upper parameter bounds.
        population: Number of candidates to construct.
        generation: Deterministic variation index for nonzero candidates.

    Returns:
        A bounded candidate tensor with shape ``(population, parameters)``.
    """
    population = int(population)
    if population < 1:
        raise ValueError("population must be positive.")
    center = theta_initial.detach().to(dtype=torch.float32).reshape(-1)
    lower = torch.minimum(theta_min.detach().to(center), theta_max.detach().to(center)).reshape(-1)
    upper = torch.maximum(theta_min.detach().to(center), theta_max.detach().to(center)).reshape(-1)
    if center.numel() != lower.numel() or center.numel() != upper.numel():
        raise ValueError("theta_initial, theta_min, and theta_max must have equal lengths.")
    finite_bounds = torch.isfinite(lower) & torch.isfinite(upper)
    safe_lower = torch.where(finite_bounds, lower, center - torch.clamp(center.abs(), min=1.0))
    safe_upper = torch.where(finite_bounds, upper, center + torch.clamp(center.abs(), min=1.0))
    center = torch.maximum(torch.minimum(center, safe_upper), safe_lower)
    rows = center.repeat(population, 1)
    if population == 1 or center.numel() == 0:
        return rows

    row_index = torch.arange(1, population, device=center.device, dtype=torch.float32).unsqueeze(1)
    column_index = torch.arange(1, center.numel() + 1, device=center.device, dtype=torch.float32).unsqueeze(0)
    phase = row_index * 1.61803398875 + column_index * 1.41421356237 + float(generation) * 0.754877666
    unit = 0.5 + 0.5 * torch.sin(phase)
    target = safe_lower.unsqueeze(0) + unit * (safe_upper - safe_lower).unsqueeze(0)
    # Stay away from extreme bounds while still producing materially different
    # physical parameters. Candidate zero remains the exact configured initial.
    rows[1:] = torch.maximum(
        torch.minimum(0.25 * center.unsqueeze(0) + 0.75 * target, safe_upper.unsqueeze(0)),
        safe_lower.unsqueeze(0),
    )
    return rows


def cuda_device_memory_snapshot(device: torch.device | str | None = None) -> DeviceMemorySnapshot:
    """Return CUDA-wide memory usage so Warp/Newton allocations are included.

    Args:
        device: Optional CUDA device whose memory should be queried.

    Returns:
        Free and total device-memory counters.
    """
    if not torch.cuda.is_available():
        return DeviceMemorySnapshot()
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except (RuntimeError, TypeError):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    return DeviceMemorySnapshot(free_bytes=int(free_bytes), total_bytes=int(total_bytes))


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sample.")
    rank = (len(ordered) - 1) * float(percentile)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _rmse(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_cpu = lhs.detach().to(device="cpu", dtype=torch.float64)
    rhs_cpu = rhs.detach().to(device="cpu", dtype=torch.float64)
    return float(torch.sqrt(torch.mean((lhs_cpu - rhs_cpu) ** 2)))


def _result_is_finite(result: Any) -> bool:
    return all(
        (value := getattr(result, name, None)) is None or bool(torch.all(torch.isfinite(value)).item())
        for name in ("positions", "velocities", "torques")
    )


def _graph_state(bridge: Any, population: int) -> tuple[int | None, bool | None]:
    cache = getattr(bridge, "_mujoco_cache", None)
    if not isinstance(cache, dict):
        return None, None
    ctx = cache.get(int(population))
    if ctx is None:
        return 0, None
    graphs = getattr(ctx, "capture_cache", None)
    return (len(graphs) if graphs is not None else None, bool(getattr(ctx, "capture_failed", False)))


def _default_synchronize(bridge: Any) -> None:
    modules = getattr(bridge, "_modules", None)
    warp = getattr(modules, "warp", None)
    synchronize = getattr(warp, "synchronize", None)
    if callable(synchronize):
        synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


async def run_population_saturation_benchmark(
    *,
    bridge: Any,
    param_entries: list[Any],
    commands: torch.Tensor,
    num_steps: int,
    duration_s_per_candidate: float,
    theta_initial: torch.Tensor,
    theta_min: torch.Tensor,
    theta_max: torch.Tensor,
    config: PopulationSaturationConfig | None = None,
    synchronize: Callable[[], None] | None = None,
    memory_probe: Callable[[], DeviceMemorySnapshot] | None = None,
) -> PopulationSaturationReport:
    """Measure steady rollout throughput while increasing candidate population.

    Args:
        bridge: Rollout bridge under benchmark.
        param_entries: Parameter mappings passed to each rollout.
        commands: Command trajectory shared by all candidates.
        num_steps: Number of simulation steps per rollout.
        duration_s_per_candidate: Simulated duration represented by each candidate.
        theta_initial: Initial parameter vector.
        theta_min: Lower parameter bounds.
        theta_max: Upper parameter bounds.
        config: Optional sweep configuration.
        synchronize: Optional device synchronization hook.
        memory_probe: Optional device-memory sampling hook.

    Returns:
        Measurements and the detected saturation point.
    """
    config = config or PopulationSaturationConfig()
    populations = config.normalized_populations()
    synchronize = synchronize or (lambda: _default_synchronize(bridge))
    memory_probe = memory_probe or cuda_device_memory_snapshot
    report = PopulationSaturationReport(
        duration_s_per_candidate=float(duration_s_per_candidate),
        num_steps=int(num_steps),
        num_parameters=int(theta_initial.numel()),
        steady_replays=int(config.steady_replays),
        capture_expected=bool(config.capture_expected),
    )
    reference_candidate0: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    population_one_median: float | None = None
    previous_throughput: float | None = None
    previous_population: int | None = None

    async def _timed(theta: torch.Tensor) -> tuple[Any, float]:
        synchronize()
        start = time.perf_counter()
        result = await bridge.run_rollout_async(theta, param_entries, commands, int(num_steps))
        synchronize()
        return result, time.perf_counter() - start

    for population in populations:
        before = memory_probe()
        sample = PopulationSaturationSample(
            population=population,
            status="running",
            device_used_before_bytes=before.used_bytes,
        )
        last_result = None
        memory_limit_reached = False
        try:
            warm_theta = build_diverse_candidate_population(
                theta_initial, theta_min, theta_max, population, generation=0
            )
            last_result, sample.warmup_s = await _timed(warm_theta)
            if config.capture_expected:
                capture_theta = build_diverse_candidate_population(
                    theta_initial, theta_min, theta_max, population, generation=1
                )
                last_result, sample.capture_build_s = await _timed(capture_theta)
            after_capture = memory_probe()
            sample.device_used_after_capture_bytes = after_capture.used_bytes

            observed_used = after_capture.used_bytes
            for replay in range(int(config.steady_replays)):
                theta = build_diverse_candidate_population(
                    theta_initial,
                    theta_min,
                    theta_max,
                    population,
                    generation=replay + 2,
                )
                last_result, elapsed = await _timed(theta)
                sample.replay_samples_s.append(elapsed)
                observed_used = max(observed_used, memory_probe().used_bytes)

            sample.finite = _result_is_finite(last_result)
            if not sample.finite:
                raise RuntimeError("non-finite rollout output")
            sample.median_s = statistics.median(sample.replay_samples_s)
            sample.p95_s = _percentile(sample.replay_samples_s, 0.95)
            sample.candidates_per_s = population / sample.median_s
            sample.wall_s_per_candidate = sample.median_s / population
            sample.aggregate_rt = population * float(duration_s_per_candidate) / sample.median_s
            if population_one_median is None and population == 1:
                population_one_median = sample.median_s
            if population_one_median is not None:
                sample.parallel_efficiency = population_one_median / sample.median_s
            if previous_throughput is not None and previous_population is not None:
                throughput_gain = sample.candidates_per_s / previous_throughput - 1.0
                sample.marginal_throughput_gain_percent = 100.0 * throughput_gain
                population_gain = population / previous_population - 1.0
                if population_gain > 0.0:
                    sample.marginal_scaling_efficiency_percent = 100.0 * throughput_gain / population_gain
            previous_throughput = sample.candidates_per_s
            previous_population = population

            current_candidate0 = (
                last_result.positions[0],
                last_result.velocities[0],
                last_result.torques[0],
            )
            if reference_candidate0 is None:
                reference_candidate0 = tuple(value.detach().cpu().clone() for value in current_candidate0)
            sample.candidate0_position_rmse = _rmse(current_candidate0[0], reference_candidate0[0])
            sample.candidate0_velocity_rmse = _rmse(current_candidate0[1], reference_candidate0[1])
            sample.candidate0_torque_rmse = _rmse(current_candidate0[2], reference_candidate0[2])
            graph_count, capture_failed = _graph_state(bridge, population)
            sample.graph_count = graph_count
            sample.capture_failed = capture_failed
            if config.capture_expected and graph_count is not None and graph_count < 1:
                raise RuntimeError("CUDA graph capture was expected but no graph is cached")

            after_replays = memory_probe()
            sample.device_used_after_replays_bytes = max(after_replays.used_bytes, observed_used)
            sample.replay_memory_drift_bytes = max(
                0,
                sample.device_used_after_replays_bytes - sample.device_used_after_capture_bytes,
            )
            peak_total_bytes = max(after_capture.total_bytes, after_replays.total_bytes)
            peak_free_bytes = max(0, peak_total_bytes - sample.device_used_after_replays_bytes)
            peak_used_fraction = (
                sample.device_used_after_replays_bytes / peak_total_bytes if peak_total_bytes > 0 else 0.0
            )
            if peak_total_bytes > 0 and (
                peak_used_fraction >= float(config.max_device_memory_fraction)
                or peak_free_bytes < int(config.minimum_free_device_bytes)
            ):
                memory_limit_reached = True
                report.stop_reason = (
                    f"device memory safety threshold reached during population {population} "
                    f"({100.0 * peak_used_fraction:.1f}% used, {peak_free_bytes} bytes free)"
                )
            sample.status = "memory_limit" if memory_limit_reached else "ok"
        except Exception as exc:  # noqa: BLE001 - an adaptive sweep must report and stop cleanly
            sample.status = "error"
            sample.error = f"{type(exc).__name__}: {exc}"
            report.stop_reason = f"population {population} failed: {sample.error}"
        finally:
            last_result = None
            synchronize()
            if config.release_contexts:
                release = getattr(bridge, "release_population_context", None)
                if callable(release):
                    release(population)
                    synchronize()
            sample.device_used_after_release_bytes = memory_probe().used_bytes
            report.samples.append(sample)

        if sample.status != "ok":
            break
        if memory_limit_reached:
            break

    successful = [sample for sample in report.samples if sample.status == "ok"]
    for index, sample in enumerate(successful[1:], start=1):
        efficiency = sample.marginal_scaling_efficiency_percent
        if efficiency is not None and efficiency < float(config.saturation_efficiency_percent):
            report.saturation_population = successful[index - 1].population
            break
    return report
