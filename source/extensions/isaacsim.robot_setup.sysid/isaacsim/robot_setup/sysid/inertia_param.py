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

"""Log-Cholesky parametrization for symmetric positive-definite inertia tensors."""

from __future__ import annotations

import numpy as np
import torch

LOG_CHOLESKY_DIM = 6


def log_cholesky_to_inertia_matrix(theta: np.ndarray) -> np.ndarray:
    """Decode unconstrained parameters to a 3x3 inertia tensor.

    Uses lower-triangular Cholesky factor ``L`` with log-diagonal entries:

    ``L = [[exp(t0), 0, 0], [t1, exp(t2), 0], [t3, t4, exp(t5)]]`` and ``I = L @ L.T``.

    Args:
        theta: Shape ``(6,)`` Log-Cholesky parameters.

    Returns:
        Inertia matrix with shape ``(3, 3)``.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(LOG_CHOLESKY_DIM)
    l11 = np.exp(theta[0])
    l22 = np.exp(theta[2])
    l33 = np.exp(theta[5])
    l21 = theta[1]
    l31 = theta[3]
    l32 = theta[4]
    lower = np.array(
        [[l11, 0.0, 0.0], [l21, l22, 0.0], [l31, l32, l33]],
        dtype=np.float64,
    )
    return lower @ lower.T


def log_cholesky_to_inertia_flat(theta: np.ndarray) -> np.ndarray:
    """Return inertia as a length-9 row-major flattening.

    Args:
        theta: Shape ``(6,)`` Log-Cholesky parameters.

    Returns:
        Row-major inertia tensor with shape ``(9,)``.
    """
    return log_cholesky_to_inertia_matrix(theta).reshape(-1)


def log_cholesky_to_inertia_matrix_torch(theta: torch.Tensor) -> torch.Tensor:
    """Batched, autograd-friendly twin of :func:`log_cholesky_to_inertia_matrix`.

    Args:
        theta: Shape ``(..., 6)`` Log-Cholesky parameters.

    Returns:
        Inertia matrices with shape ``(..., 3, 3)``, graph-connected to ``theta``.
    """
    if theta.shape[-1] != LOG_CHOLESKY_DIM:
        raise ValueError(f"Log-Cholesky theta must have trailing dimension {LOG_CHOLESKY_DIM}, got {theta.shape}.")
    zero = torch.zeros_like(theta[..., 0])
    rows = (
        torch.stack((torch.exp(theta[..., 0]), zero, zero), dim=-1),
        torch.stack((theta[..., 1], torch.exp(theta[..., 2]), zero), dim=-1),
        torch.stack((theta[..., 3], theta[..., 4], torch.exp(theta[..., 5])), dim=-1),
    )
    lower = torch.stack(rows, dim=-2)
    return lower @ lower.transpose(-1, -2)


def inertia_matrix_to_log_cholesky(inertia: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Encode a 3x3 inertia tensor into Log-Cholesky parameters.

    Args:
        inertia: Shape ``(3, 3)`` or ``(9,)`` row-major symmetric matrix.
        eps: Positive diagonal regularization and logarithm floor.

    Returns:
        Log-Cholesky vector with shape ``(6,)``.
    """
    matrix = np.asarray(inertia, dtype=np.float64)
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3):
        raise ValueError(f"Inertia must be 3x3 or length 9, got shape {matrix.shape}.")
    sym = 0.5 * (matrix + matrix.T)
    sym.flat[::4] += eps
    try:
        lower = np.linalg.cholesky(sym)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Inertia matrix is not positive definite.") from exc
    return np.array(
        [
            np.log(max(lower[0, 0], eps)),
            lower[1, 0],
            np.log(max(lower[1, 1], eps)),
            lower[2, 0],
            lower[2, 1],
            np.log(max(lower[2, 2], eps)),
        ],
        dtype=np.float64,
    )


def log_cholesky_jacobian(theta: np.ndarray) -> np.ndarray:
    """Jacobian ``d(flatten(I)) / d(theta)`` with shape ``(9, 6)``.

    Args:
        theta: Shape ``(6,)`` Log-Cholesky parameters.

    Returns:
        Analytic row-major inertia Jacobian with shape ``(9, 6)``.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(LOG_CHOLESKY_DIM)
    l11 = np.exp(theta[0])
    l22 = np.exp(theta[2])
    l33 = np.exp(theta[5])
    l21 = theta[1]
    l31 = theta[3]
    l32 = theta[4]

    # I = L L^T; flatten row-major.
    d_i_flat_d_theta = np.zeros((9, LOG_CHOLESKY_DIM), dtype=np.float64)

    def set_flat(row: int, col: int, derivs: dict[int, float]) -> None:
        idx = row * 3 + col
        for t_idx, val in derivs.items():
            d_i_flat_d_theta[idx, t_idx] += val

    # Diagonal entries
    set_flat(0, 0, {0: 2.0 * l11 * l11})
    set_flat(1, 1, {1: 2.0 * l21, 2: 2.0 * l22 * l22})
    set_flat(2, 2, {3: 2.0 * l31, 4: 2.0 * l32, 5: 2.0 * l33 * l33})

    # Off-diagonal (symmetric contributions)
    set_flat(0, 1, {0: l11 * l21, 1: l11})
    set_flat(1, 0, {0: l11 * l21, 1: l11})
    set_flat(0, 2, {0: l11 * l31, 3: l11})
    set_flat(2, 0, {0: l11 * l31, 3: l11})
    set_flat(1, 2, {1: l31, 2: l22 * l32, 3: l21, 4: l22})
    set_flat(2, 1, {1: l31, 2: l22 * l32, 3: l21, 4: l22})

    return d_i_flat_d_theta
