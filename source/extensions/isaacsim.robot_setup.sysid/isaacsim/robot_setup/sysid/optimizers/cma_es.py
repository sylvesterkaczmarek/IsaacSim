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


"""CMA-ES optimizer for derivative-free system identification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import torch

from ..env_bridge import SysIdEnvironmentBridgeError
from ..optimizer_base import OptimizationIterationStatus, OptimizerConfig
from ..optimizer_config import OptimizerBackendConfig
from ..parameter_space import ParameterSpace
from ..rollout_optimizer_base import RolloutOptimizerBase
from ..runtime import yield_control
from .cma_es_core import (
    CmaEsState,
    default_population_size,
    sample_mirrored_population,
    tell,
)
from .cost_utils import penalize_nonfinite_costs
from .parameter_transform import BoxParameterTransform


class CmaEsOptimizer(RolloutOptimizerBase):
    """Covariance Matrix Adaptation Evolution Strategy over rollout costs."""

    progress_label = "CMA-ES"

    def __init__(self) -> None:
        super().__init__()
        self._parameter_space: Optional[ParameterSpace] = None

    def set_parameter_space(self, parameter_space: ParameterSpace | None) -> None:  # noqa: D102
        self._parameter_space = parameter_space

    async def run_async(  # noqa: D102
        self,
        config: OptimizerConfig,
        on_iteration: Optional[Callable[[OptimizationIterationStatus], None]] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> OptimizationIterationStatus:
        self.reset_cancel()
        device = self._bridge.device
        m = len(config.param_entries)
        if m < 1:
            raise ValueError("Select at least one parameter to optimize.")

        backend_cfg = config.backend_config or OptimizerBackendConfig()
        theta_min = config.theta_min.to(device=device, dtype=torch.float32)
        theta_max = config.theta_max.to(device=device, dtype=torch.float32)
        theta = config.theta_initial.to(device=device, dtype=torch.float32).clone()

        transform = BoxParameterTransform.from_bounds(theta_min, theta_max)
        mean_norm = np.clip(transform.physical_to_normalized(theta.detach().cpu().numpy()), 0.0, 1.0)
        fallback_norm = None
        if config.theta_fallback is not None:
            # Guard against a harmful analytical-presolve seed: the pre-presolve
            # initials compete in the first generation, so `tell` and the best-theta
            # tracking see both starts and keep whichever is better at rollout cost.
            fallback_norm = np.clip(
                transform.physical_to_normalized(
                    config.theta_fallback.to(device=device, dtype=torch.float32).detach().cpu().numpy()
                ),
                0.0,
                1.0,
            )
        population_size = backend_cfg.cma_population_size or default_population_size(m)
        population_size = max(4, int(population_size))
        batch_size = _resolve_cma_batch_size(backend_cfg, population_size)
        cma_seed = int(getattr(backend_cfg, "cma_seed", 0))
        if cma_seed < 0:
            raise ValueError("cma_seed must be non-negative.")
        rng = np.random.default_rng(cma_seed)
        state = CmaEsState.create(m, initial_mean=mean_norm, sigma=float(backend_cfg.cma_sigma))

        best_theta = theta.clone()
        best_norm = mean_norm.copy()
        best_cost = float("inf")
        best_plot_data = None
        status = OptimizationIterationStatus(
            iteration=0,
            cost=best_cost,
            damping=state.sigma,
            accepted=True,
            theta=best_theta.detach().cpu().tolist(),
        )

        max_generations = config.max_iterations
        for generation in range(1, max_generations + 1):
            self._raise_if_cancelled()

            gen_start = (generation - 1) / max(max_generations, 1)
            gen_end = generation / max(max_generations, 1)
            if on_progress:
                on_progress(f"{self.progress_label} generation {generation}/{max_generations}", gen_start)
                await yield_control()

            candidates_norm = sample_mirrored_population(state, population_size, rng)
            candidates_norm[0] = state.mean
            if population_size > 1 and best_cost < float("inf"):
                candidates_norm[1] = best_norm
            elif population_size > 1 and generation == 1 and fallback_norm is not None:
                candidates_norm[1] = fallback_norm
            candidates_norm = np.clip(candidates_norm, 0.0, 1.0)

            costs, plot_data = await self._evaluate_population_in_batches(
                config,
                transform,
                candidates_norm,
                batch_size=batch_size,
                device=device,
                on_progress=self._progress_for_phase(on_progress, gen_start, gen_end),
            )
            costs_np = costs.detach().cpu().numpy()
            costs_for_update, finite_mask = penalize_nonfinite_costs(costs_np)
            accepted = False
            if np.any(finite_mask):
                state = tell(state, candidates_norm, costs_for_update, population_size=population_size)

                gen_best_idx = int(np.argmin(costs_for_update))
                gen_best_cost = float(costs_np[gen_best_idx])
                gen_best_norm = candidates_norm[gen_best_idx]
                gen_best_theta = transform.normalized_to_theta_batch(
                    gen_best_norm,
                    device=device,
                    parameter_space=self._parameter_space,
                    param_entries=config.param_entries,
                )[0]
                accepted = bool(finite_mask[gen_best_idx]) and gen_best_cost < best_cost
            else:
                gen_best_idx = -1
                gen_best_cost = float("inf")
                gen_best_norm = best_norm
                gen_best_theta = best_theta
            if accepted:
                best_cost = gen_best_cost
                best_theta = gen_best_theta.clone()
                best_norm = gen_best_norm.copy()
                if gen_best_idx == 0 and plot_data is not None:
                    best_plot_data = plot_data
                else:
                    # Keep the MuJoCo world count fixed for the entire solve so
                    # the population graph remains reusable. The final replay
                    # below refreshes plot data with a repeated best candidate.
                    best_plot_data = None

            status = OptimizationIterationStatus(
                iteration=generation,
                cost=best_cost,
                damping=float(state.sigma),
                accepted=accepted,
                theta=best_theta.detach().cpu().tolist(),
                rollout=best_plot_data,
            )
            await yield_control()
            if on_iteration is not None:
                on_iteration(status)
            if on_progress:
                on_progress(f"{self.progress_label} generation {generation}/{max_generations}: complete", gen_end)
            await yield_control()

        self._raise_if_cancelled()
        if not np.isfinite(best_cost):
            raise SysIdEnvironmentBridgeError(
                "CMA-ES did not produce a finite rollout cost. "
                "Check trajectory values, parameter bounds, and simulation stability."
            )
        if best_cost < float("inf"):
            if on_progress:
                on_progress(f"{self.progress_label} final best rollout", 0.99)
                await yield_control()
            # Preserve the generation batch size so parallel bridges can reuse
            # their population-specific graph/context for the final plot replay.
            # Only row zero is consumed; every row intentionally carries best_theta.
            final_theta_batch = best_theta.unsqueeze(0).repeat(batch_size, 1)
            final_costs, _, final_plot = await self.compute_costs_for_theta_batch(
                config,
                final_theta_batch,
                on_progress=self._progress_for_phase(on_progress, 0.99, 1.0),
            )
            final_cost = float(final_costs[0].item())
            if np.isfinite(final_cost):
                best_cost = final_cost
                best_plot_data = final_plot
            status = OptimizationIterationStatus(
                iteration=status.iteration,
                cost=best_cost,
                damping=float(state.sigma),
                accepted=status.accepted,
                theta=best_theta.detach().cpu().tolist(),
                rollout=best_plot_data,
            )

        self._raise_if_cancelled()
        if on_progress:
            on_progress("Optimization finished", 1.0)
        return status

    async def _evaluate_population_in_batches(
        self,
        config: OptimizerConfig,
        transform: BoxParameterTransform,
        candidates_norm: np.ndarray,
        *,
        batch_size: int,
        device: torch.device,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> tuple[torch.Tensor, object | None]:
        population_size = int(candidates_norm.shape[0])
        batch_size = max(1, min(int(batch_size), population_size))
        total_batches = int(np.ceil(population_size / batch_size))
        costs_by_batch: list[torch.Tensor] = []
        plot_data = None

        for batch_index, start in enumerate(range(0, population_size, batch_size)):
            end = min(population_size, start + batch_size)
            theta_batch = transform.normalized_to_theta_batch(
                candidates_norm[start:end],
                device=device,
                parameter_space=self._parameter_space,
                param_entries=config.param_entries,
            )

            def batch_progress(message: str, fraction: float, *, _index=batch_index) -> None:  # noqa: ANN001
                if on_progress is None:
                    return
                local = max(0.0, min(1.0, float(fraction)))
                on_progress(
                    f"batch {_index + 1}/{total_batches}: {message}",
                    (_index + local) / max(1, total_batches),
                )

            costs, _, batch_plot = await self.compute_costs_for_theta_batch(
                config,
                theta_batch,
                on_progress=batch_progress if on_progress is not None else None,
                include_plot_data=(plot_data is None),
            )
            costs_by_batch.append(costs)
            if plot_data is None and batch_plot is not None:
                plot_data = batch_plot

        return torch.cat(costs_by_batch, dim=0), plot_data


def _resolve_cma_batch_size(backend_cfg: OptimizerBackendConfig, population_size: int) -> int:
    raw = getattr(backend_cfg, "cma_batch_size", None)
    if raw is None:
        return population_size
    return max(1, min(int(raw), int(population_size)))
