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

"""Shared candidate transforms for derivative-free optimizer backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..lm_solver import clamp_parameters
from ..parameter_space import ParameterSpace
from ..parameter_types import SysIdParameterEntry


def _normalize_bounds(
    theta_min: np.ndarray,
    theta_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.minimum(theta_min, theta_max)
    upper = np.maximum(theta_min, theta_max)
    span = np.maximum(upper - lower, 1e-12)
    return lower, upper, span


def physical_to_normalized(theta: np.ndarray, lower: np.ndarray, span: np.ndarray) -> np.ndarray:  # noqa: D103
    return (theta - lower) / span


def normalized_to_physical(normalized: np.ndarray, lower: np.ndarray, span: np.ndarray) -> np.ndarray:  # noqa: D103
    return lower + normalized * span


@dataclass(frozen=True)
class BoxParameterTransform:
    """Map between normalized optimizer candidates and clamped physical theta tensors."""

    lower: np.ndarray
    upper: np.ndarray
    span: np.ndarray
    theta_min: torch.Tensor
    theta_max: torch.Tensor

    @classmethod
    def from_bounds(cls, theta_min: torch.Tensor, theta_max: torch.Tensor) -> BoxParameterTransform:  # noqa: D102
        lower, upper, span = _normalize_bounds(
            theta_min.detach().cpu().numpy(),
            theta_max.detach().cpu().numpy(),
        )
        return cls(lower=lower, upper=upper, span=span, theta_min=theta_min, theta_max=theta_max)

    def physical_to_normalized(self, theta: np.ndarray) -> np.ndarray:  # noqa: D102
        return physical_to_normalized(theta, self.lower, self.span)

    def normalized_to_physical(self, normalized: np.ndarray) -> np.ndarray:  # noqa: D102
        return normalized_to_physical(normalized, self.lower, self.span)

    def normalized_to_theta_batch(  # noqa: D102
        self,
        normalized: np.ndarray,
        *,
        device: torch.device,
        parameter_space: ParameterSpace | None = None,
        param_entries: list[SysIdParameterEntry] | None = None,
    ) -> torch.Tensor:
        candidates = np.asarray(normalized, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)

        theta_batch = torch.tensor(
            self.normalized_to_physical(candidates),
            device=device,
            dtype=torch.float32,
        )
        if parameter_space is not None:
            if param_entries is None:
                raise ValueError("param_entries are required when applying ParameterSpace constraints.")
            for row_i in range(theta_batch.shape[0]):
                theta_batch[row_i] = parameter_space.reparameterize_inertia_theta(theta_batch[row_i], param_entries)
        return clamp_parameters(
            theta_batch,
            self.theta_min.to(device=theta_batch.device, dtype=theta_batch.dtype),
            self.theta_max.to(device=theta_batch.device, dtype=theta_batch.dtype),
        )
