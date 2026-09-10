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


"""Numpy CMA-ES core (Hansen 2016, mu/mu_w, lambda)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def default_population_size(dimension: int) -> int:  # noqa: D103
    return max(4, 4 + int(3 * np.log(max(dimension, 1))))


@dataclass
class CmaEsState:
    """Internal CMA-ES state in normalized [0, 1] box coordinates."""

    dimension: int
    mean: np.ndarray
    sigma: float
    covariance: np.ndarray
    evolution_path_sigma: np.ndarray
    evolution_path_c: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    inverse_sqrt_covariance: np.ndarray
    generation: int = 0

    @classmethod
    def create(cls, dimension: int, *, initial_mean: np.ndarray, sigma: float) -> CmaEsState:  # noqa: D102
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        mean = np.asarray(initial_mean, dtype=np.float64).reshape(-1)
        if mean.shape[0] != dimension:
            raise ValueError(f"initial_mean length {mean.shape[0]} != dimension {dimension}")
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be a finite positive number.")
        identity = np.eye(dimension, dtype=np.float64)
        eigenvalues = np.ones(dimension, dtype=np.float64)
        return cls(
            dimension=dimension,
            mean=mean.copy(),
            sigma=sigma,
            covariance=identity.copy(),
            evolution_path_sigma=np.zeros(dimension, dtype=np.float64),
            evolution_path_c=np.zeros(dimension, dtype=np.float64),
            eigenvalues=eigenvalues,
            eigenvectors=identity.copy(),
            inverse_sqrt_covariance=identity.copy(),
        )


def _compute_weights(population_size: int) -> tuple[int, np.ndarray, float]:
    mu = population_size // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)
    return mu, weights, mueff


def sample_mirrored_population(state: CmaEsState, population_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample candidates in mirrored pairs around the current mean.

    Args:
        state: Current CMA-ES distribution state.
        population_size: Number of candidates in the CMA-ES generation.
        rng: Random-number generator used for candidate sampling.

    Returns:
        Candidate matrix with shape ``(population_size, state.dimension)``.
    """
    pair_count = population_size // 2
    extra_count = population_size - pair_count * 2
    z_half = rng.standard_normal((pair_count, state.dimension))
    z = np.concatenate([z_half, -z_half], axis=0)
    if extra_count:
        z = np.concatenate([z, rng.standard_normal((extra_count, state.dimension))], axis=0)
    y = (z * np.sqrt(state.eigenvalues)) @ state.eigenvectors.T
    return state.mean + state.sigma * y


def tell(
    state: CmaEsState,
    candidates: np.ndarray,
    costs: np.ndarray,
    *,
    population_size: int,
) -> CmaEsState:
    """Update CMA-ES state from evaluated population costs.

    Args:
        state: Current CMA-ES distribution state.
        candidates: Evaluated candidate coordinates.
        costs: Scalar objective values for evaluated candidates.
        population_size: Number of candidates in the CMA-ES generation.

    Returns:
        The updated CMA-ES distribution state.
    """
    state = replace(
        state,
        mean=state.mean.copy(),
        covariance=state.covariance.copy(),
        evolution_path_sigma=state.evolution_path_sigma.copy(),
        evolution_path_c=state.evolution_path_c.copy(),
        eigenvalues=state.eigenvalues.copy(),
        eigenvectors=state.eigenvectors.copy(),
        inverse_sqrt_covariance=state.inverse_sqrt_covariance.copy(),
    )
    dim = state.dimension
    mu, weights, mueff = _compute_weights(population_size)
    order = np.argsort(costs)
    selected = candidates[order[:mu]]

    mean_old = state.mean.copy()
    state.mean = weights @ selected

    # Scaled mutation vectors: y_k = (x_k - m_old) / sigma
    y_k = (selected - mean_old) / state.sigma
    y_w = weights @ y_k

    # Hyperparameters tuned relative to mueff instead of the integer mu
    c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
    c_c = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c_1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    c_mu = min(1.0 - c_1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
    chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))

    # Isotropic coordinate transformation for evolution path sigma
    invsqrt_C_yw = state.inverse_sqrt_covariance @ y_w
    state.evolution_path_sigma = (1.0 - c_sigma) * state.evolution_path_sigma + np.sqrt(
        c_sigma * (2.0 - c_sigma) * mueff
    ) * invsqrt_C_yw

    h_sigma = np.linalg.norm(state.evolution_path_sigma) / np.sqrt(
        1.0 - (1.0 - c_sigma) ** (2.0 * (state.generation + 1))
    ) / chi_n < 1.4 + 2.0 / (dim + 1.0)

    state.evolution_path_c = (1.0 - c_c) * state.evolution_path_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mueff) * y_w

    # Compute updates
    rank_one = np.outer(state.evolution_path_c, state.evolution_path_c)
    if not h_sigma:
        rank_one += c_c * (2.0 - c_c) * state.covariance
    # Vectorized rank-mu computation
    rank_mu = (y_k.T * weights) @ y_k

    # Covariance and Step size updates
    state.covariance = (1.0 - c_1 - c_mu) * state.covariance + c_1 * rank_one + c_mu * rank_mu
    state.sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(state.evolution_path_sigma) / chi_n - 1.0))
    state.sigma = float(np.clip(state.sigma, 1e-12, 1.0))

    # Numerical stabilization & Eigendecomposition
    state.covariance = 0.5 * (state.covariance + state.covariance.T)
    state.eigenvalues, state.eigenvectors = np.linalg.eigh(state.covariance)
    state.eigenvalues = np.maximum(state.eigenvalues, 1e-12)
    state.inverse_sqrt_covariance = (
        state.eigenvectors @ np.diag(1.0 / np.sqrt(state.eigenvalues)) @ state.eigenvectors.T
    )
    state.generation += 1
    return state
