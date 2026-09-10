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

"""Levenberg-Marquardt helpers: finite-difference Jacobian and linear solve."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .parameter_space import ParameterSpace
    from .parameter_types import SysIdParameterEntry


def compute_residual_cost(r_nominal: torch.Tensor) -> torch.Tensor:
    """Scalar cost 0.5 * ||r||^2 for the nominal residual vector.

    Args:
        r_nominal: Shape (R,) residual vector.

    Returns:
        Scalar tensor cost.
    """
    return 0.5 * torch.dot(r_nominal, r_nominal)


def compute_jacobian_finite_difference(
    r_nominal: torch.Tensor,
    r_perturbed: torch.Tensor,
    epsilon: float,
    column_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build Jacobian J with shape (R, M) via forward differences.

    ``J[:, i] = (r_perturbed[i] - r_nominal) / epsilon``

    Args:
        r_nominal: Shape (R,) residual at nominal parameters (env 0).
        r_perturbed: Shape (M, R) residuals at perturbed clones (envs 1..M).
        epsilon: Default finite-difference step size (must be > 0 when column_steps is not provided).
        column_steps: Optional actual perturbation per parameter. Zero-width columns produce zero Jacobian columns.

    Returns:
        Jacobian tensor of shape (R, M).
    """
    if column_steps is None and epsilon <= 0.0:
        raise ValueError("epsilon must be positive for finite-difference Jacobian.")

    # r_perturbed[i] corresponds to perturbation of parameter i
    diff = r_perturbed - r_nominal.unsqueeze(0)
    if column_steps is not None:
        steps = column_steps.to(device=diff.device, dtype=diff.dtype).reshape(-1)
        if steps.shape[0] != diff.shape[0]:
            raise ValueError(f"column_steps length ({steps.shape[0]}) must match perturbation count ({diff.shape[0]}).")
        jacobian_t = torch.zeros_like(diff)
        valid = torch.abs(steps) > torch.finfo(diff.dtype).eps
        if torch.any(valid):
            jacobian_t[valid] = diff[valid] / steps[valid].unsqueeze(1)
        return jacobian_t.T.contiguous()
    # (M, R) -> (R, M)
    return (diff / epsilon).T.contiguous()


def compute_jacobian(
    r_nominal: torch.Tensor,
    r_perturbed: torch.Tensor | None,
    epsilon: float,
    *,
    analytical_jacobian: Optional[torch.Tensor] = None,
    parameter_count: int | None = None,
    column_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build Jacobian ``(R, M)`` using analytical or finite-difference columns (REQ-22).

    Args:
        r_nominal: Residual vector at the nominal parameter row.
        r_perturbed: Residual rows evaluated at perturbed parameters.
        epsilon: Finite-difference perturbation magnitude.
        analytical_jacobian: Optional precomputed residual Jacobian.
        parameter_count: Expected number of Jacobian columns.
        column_steps: Actual finite-difference displacement for each parameter.

    Returns:
        Residual Jacobian with shape ``(R, M)``.
    """
    if analytical_jacobian is not None:
        expected_columns = parameter_count
        if expected_columns is None:
            expected_columns = r_perturbed.shape[0] if r_perturbed is not None else analytical_jacobian.shape[1]
        if analytical_jacobian.shape[1] != expected_columns:
            raise ValueError(
                f"Analytical Jacobian columns ({analytical_jacobian.shape[1]}) must match "
                f"parameter count ({expected_columns})."
            )
        analytical = analytical_jacobian.to(device=r_nominal.device, dtype=r_nominal.dtype)
        if analytical.shape[0] != r_nominal.shape[0]:
            raise ValueError(
                f"Analytical Jacobian rows ({analytical.shape[0]}) must match "
                f"residual dimension ({r_nominal.shape[0]})."
            )
        return analytical
    if r_perturbed is None:
        raise ValueError("r_perturbed is required when analytical_jacobian is not provided.")
    return compute_jacobian_finite_difference(r_nominal, r_perturbed, epsilon, column_steps=column_steps)


def levenberg_marquardt_step(
    jacobian: torch.Tensor,
    r_nominal: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """Solve (J^T J + lambda I) delta = -J^T r.

    Args:
        jacobian: Shape (R, M) Jacobian matrix.
        r_nominal: Shape (R,) nominal residual.
        damping: Levenberg-Marquardt lambda (>= 0).

    Returns:
        Parameter update delta of shape (M,).
    """
    m = jacobian.shape[1]
    jtj = jacobian.T @ jacobian
    # Keep the classic LM lambda-I damping used by existing benchmark baselines.
    # Scale-invariant diagonal damping is a separate algorithm policy because it
    # changes convergence behavior and requires retuning the configured lambda.
    identity = torch.eye(m, device=jacobian.device, dtype=jacobian.dtype)
    lhs = jtj + damping * identity
    rhs = -(jacobian.T @ r_nominal)
    try:
        return torch.linalg.solve(lhs, rhs)
    except RuntimeError as solve_error:
        try:
            return torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)
        except RuntimeError as lstsq_error:
            raise RuntimeError(
                "Levenberg-Marquardt linear solve and least-squares fallback both failed: "
                f"solve={solve_error}; lstsq={lstsq_error}"
            ) from lstsq_error


def clamp_parameters(
    theta: torch.Tensor,
    theta_min: torch.Tensor,
    theta_max: torch.Tensor,
) -> torch.Tensor:
    """Clamp parameter vector to box bounds.

    Args:
        theta: Shape (M,).
        theta_min: Shape (M,) lower bounds.
        theta_max: Shape (M,) upper bounds.

    Returns:
        Clamped copy of theta.
    """
    lower = torch.minimum(theta_min, theta_max)
    upper = torch.maximum(theta_min, theta_max)
    return torch.clamp(theta, min=lower, max=upper)


def build_bounded_perturbed_theta_batch(
    theta: torch.Tensor,
    epsilon: float,
    theta_min: torch.Tensor,
    theta_max: torch.Tensor,
    *,
    parameter_space: "ParameterSpace | None" = None,
    param_entries: list["SysIdParameterEntry"] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build bounded one-at-a-time perturbations and return actual per-column steps.

    The nominal row is unchanged. Each perturbed row prefers a positive step, falls
    back to a negative step near an upper bound, and stays unchanged for fixed-width
    parameters.

    Args:
        theta: Nominal parameter vector with shape ``(M,)``.
        epsilon: Finite-difference perturbation magnitude.
        theta_min: Lower box bound for each parameter.
        theta_max: Upper box bound for each parameter.
        parameter_space: Optional registry used to enforce coupled constraints.
        param_entries: Selected parameter entries in theta order.

    Returns:
        Perturbation batch with shape ``(M + 1, M)`` and the actual step for each column.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive for finite-difference perturbations.")
    if parameter_space is not None and param_entries is None:
        raise ValueError("param_entries are required when applying ParameterSpace constraints.")

    theta_min = theta_min.to(device=theta.device, dtype=theta.dtype)
    theta_max = theta_max.to(device=theta.device, dtype=theta.dtype)
    lower = torch.minimum(theta_min, theta_max)
    upper = torch.maximum(theta_min, theta_max)
    nominal = clamp_parameters(theta, lower, upper)
    m = nominal.shape[0]
    batch = nominal.unsqueeze(0).expand(m + 1, m).clone()
    steps = torch.zeros(m, device=theta.device, dtype=theta.dtype)
    eps_tensor = torch.tensor(float(epsilon), device=theta.device, dtype=theta.dtype)

    for i in range(m):
        positive_room = upper[i] - nominal[i]
        negative_room = nominal[i] - lower[i]
        step = torch.minimum(eps_tensor, positive_room)
        if torch.abs(step) <= torch.finfo(theta.dtype).eps:
            step = -torch.minimum(eps_tensor, negative_room)
        if torch.abs(step) <= torch.finfo(theta.dtype).eps:
            continue

        row = nominal.clone()
        row[i] = row[i] + step
        row = clamp_parameters(row, lower, upper)
        if parameter_space is not None:
            intended_row = row
            repaired_row = parameter_space.reparameterize_inertia_theta(row, param_entries)
            repaired_row = clamp_parameters(repaired_row, lower, upper)
            tolerance = torch.finfo(theta.dtype).eps * 8.0
            if not torch.allclose(repaired_row, intended_row, rtol=0.0, atol=tolerance):
                continue
            row = repaired_row
        batch[i + 1] = row
        steps[i] = row[i] - nominal[i]

    return batch, steps
