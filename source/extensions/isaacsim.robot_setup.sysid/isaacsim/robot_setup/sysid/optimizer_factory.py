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

"""Factory for SysId optimizer backends."""

from __future__ import annotations

from .optimizer_base import Optimizer
from .optimizer_config import OptimizerBackend
from .optimizers.bayesian import BayesianOptimizer
from .optimizers.cma_es import CmaEsOptimizer
from .optimizers.gradient_descent import GradientDescentOptimizer
from .optimizers.levenberg_marquardt import LevenbergMarquardtOptimizer


def create_optimizer(backend: OptimizerBackend) -> Optimizer:
    """Instantiate an optimizer for the selected backend.

    Args:
        backend: Resolved concrete optimizer backend.

    Returns:
        Concrete optimizer implementation.
    """
    if backend == OptimizerBackend.CMA_ES:
        return CmaEsOptimizer()
    if backend == OptimizerBackend.BAYESIAN:
        return BayesianOptimizer()
    if backend == OptimizerBackend.GRADIENT_DESCENT:
        return GradientDescentOptimizer()
    if backend == OptimizerBackend.LEVENBERG_MARQUARDT:
        return LevenbergMarquardtOptimizer()
    raise ValueError(f"Optimizer backend must be resolved to a supported concrete backend; got {backend!r}.")
