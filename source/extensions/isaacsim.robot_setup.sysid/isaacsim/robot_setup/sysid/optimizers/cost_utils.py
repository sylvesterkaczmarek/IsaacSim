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

"""Small cost-array helpers shared by derivative-free optimizers."""

from __future__ import annotations

import numpy as np


def finite_cost_mask(costs: np.ndarray) -> np.ndarray:
    """Return a flat boolean mask for finite scalar costs.

    Args:
        costs: Scalar objective values for evaluated candidates.

    Returns:
        Finite-cost mask or penalized cost output.
    """
    return np.isfinite(np.asarray(costs, dtype=np.float64).reshape(-1))


def penalize_nonfinite_costs(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace non-finite costs with a large finite penalty for ranking.

    Returns the penalized cost vector and the original finite mask. When no finite
    cost exists, the returned vector is unchanged so callers can skip model updates.

    Args:
        costs: Scalar objective values for evaluated candidates.

    Returns:
        Finite-cost mask or penalized cost output.
    """
    values = np.asarray(costs, dtype=np.float64).reshape(-1)
    finite = finite_cost_mask(values)
    if not np.any(finite):
        return values.copy(), finite
    penalized = values.copy()
    finite_values = values[finite]
    max_finite = float(np.max(finite_values))
    penalty = max(1.0, abs(max_finite) * 10.0)
    penalized[~finite] = penalty
    return penalized, finite
