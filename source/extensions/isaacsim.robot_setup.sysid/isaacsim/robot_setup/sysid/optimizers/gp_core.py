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


"""Expected-improvement acquisition for GP Bayesian optimization."""

from __future__ import annotations

import math

import numpy as np


def standardize_observations(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Center and scale finite observations for a unit-scale GP prior.

    Args:
        values: One-dimensional finite observations.

    Returns:
        Standardized values, original mean, and scale.
    """
    observations = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(observations))
    scale = float(np.std(observations))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        scale = 1.0
    return (observations - mean) / scale, mean, scale


def rbf_kernel(  # noqa: D103
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    length_scale: float = 0.2,
    signal_variance: float = 1.0,
) -> np.ndarray:
    diff = x_a[:, None, :] - x_b[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    return signal_variance * np.exp(-0.5 * sq_dist / max(length_scale**2, 1e-12))


def gp_posterior(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    *,
    length_scale: float = 0.2,
    signal_variance: float = 1.0,
    noise_variance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return GP posterior mean and std at query points.

    Args:
        x_train: Normalized observed coordinates.
        y_train: Observed standardized objective values.
        x_query: Normalized query coordinates.
        length_scale: RBF kernel length scale.
        signal_variance: RBF kernel signal variance.
        noise_variance: Diagonal observation-noise variance.

    Returns:
        Posterior statistics or expected-improvement scores.
    """
    k = rbf_kernel(x_train, x_train, length_scale=length_scale, signal_variance=signal_variance)
    k_s = rbf_kernel(x_train, x_query, length_scale=length_scale, signal_variance=signal_variance)

    identity = np.eye(k.shape[0])
    jitter = max(float(noise_variance), np.finfo(np.float64).eps)
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(k + jitter * identity)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise np.linalg.LinAlgError("GP covariance remained non-positive-definite after adaptive jitter.")

    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
    mu = k_s.T @ alpha
    v = np.linalg.solve(chol, k_s)
    prior_variance = np.full(x_query.shape[0], signal_variance + noise_variance, dtype=np.float64)
    var = np.clip(prior_variance - np.sum(v * v, axis=0), 0.0, None)
    return mu, np.sqrt(var)


def expected_improvement_numpy(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_observed: float,
    *,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected improvement acquisition (minimization, numpy-only).

    Args:
        mu: Posterior mean values at query points.
        sigma: Posterior standard deviations at query points.
        best_observed: Lowest observed objective value.
        xi: Expected-improvement exploration offset.

    Returns:
        Posterior statistics or expected-improvement scores.
    """
    improvement = best_observed - mu - xi
    z = improvement / np.maximum(sigma, 1e-12)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    erf_values = np.fromiter(
        (math.erf(float(value) / math.sqrt(2.0)) for value in z.flat),
        dtype=np.float64,
        count=z.size,
    ).reshape(z.shape)
    cdf = 0.5 * (1.0 + erf_values)
    ei = improvement * cdf + sigma * pdf
    ei[sigma <= np.sqrt(np.finfo(np.float64).eps)] = 0.0
    return ei
