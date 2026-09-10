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

"""Optimizer backend selection and persisted settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .actuator_compatibility import is_actuator_delay_parameter
from .parameter_types import SysIdParameterEntry
from .residual_config import ResidualWeightConfig

HIGH_DIMENSION_THRESHOLD = 8


class OptimizerBackend(str, Enum):
    """Available optimizer backend implementations."""

    AUTO = "auto"
    LEVENBERG_MARQUARDT = "levenberg_marquardt"
    CMA_ES = "cma_es"
    BAYESIAN = "bayesian"
    GRADIENT_DESCENT = "gradient_descent"


OPTIMIZER_BACKEND_LABELS: tuple[str, ...] = (
    "Auto",
    "Levenberg-Marquardt",
    "CMA-ES",
    "Bayesian",
    "Gradient Descent (Adam)",
)

_BACKEND_FROM_LABEL = {
    "Auto": OptimizerBackend.AUTO,
    "Levenberg-Marquardt": OptimizerBackend.LEVENBERG_MARQUARDT,
    "CMA-ES": OptimizerBackend.CMA_ES,
    "Bayesian": OptimizerBackend.BAYESIAN,
    "Gradient Descent (Adam)": OptimizerBackend.GRADIENT_DESCENT,
}

_BACKEND_TO_LABEL = {value: label for label, value in _BACKEND_FROM_LABEL.items()}


@dataclass
class OptimizerBackendConfig:
    """Backend-specific hyperparameters."""

    backend: OptimizerBackend = OptimizerBackend.AUTO
    cma_population_size: int | None = None
    cma_sigma: float = 0.3
    cma_batch_size: int | None = None
    cma_seed: int = 0
    bo_initial_samples: int = 5
    bo_candidate_count: int = 64
    bo_batch_size: int = 4
    bo_seed: int = 0
    gd_learning_rate: float = 0.05
    gd_num_restarts: int = 1
    gd_seed: int = 0
    use_analytical_jacobian: bool = False
    use_analytical_presolve: bool = False
    # Set by the run controller when the active bridge exposes autograd rollouts;
    # steers auto backend selection toward gradient descent.
    differentiable_bridge: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OptimizerBackendConfig:
        """Create backend settings from a JSON/YAML payload.

        Args:
            payload: Serialized input payload.

        Returns:
            Validated backend configuration using defaults for omitted fields.
        """
        backend_raw = payload.get("backend", OptimizerBackend.AUTO.value)
        backend = OptimizerBackend(backend_raw)
        return cls(
            backend=backend,
            cma_population_size=payload.get("cma_population_size"),
            cma_sigma=float(payload.get("cma_sigma", 0.3)),
            cma_batch_size=payload.get("cma_batch_size"),
            cma_seed=int(payload.get("cma_seed", 0)),
            bo_initial_samples=int(payload.get("bo_initial_samples", 5)),
            bo_candidate_count=int(payload.get("bo_candidate_count", 64)),
            bo_batch_size=int(payload.get("bo_batch_size", 4)),
            bo_seed=int(payload.get("bo_seed", 0)),
            gd_learning_rate=float(payload.get("gd_learning_rate", 0.05)),
            gd_num_restarts=int(payload.get("gd_num_restarts", 1)),
            gd_seed=int(payload.get("gd_seed", 0)),
            use_analytical_jacobian=bool(payload.get("use_analytical_jacobian", False)),
            use_analytical_presolve=bool(payload.get("use_analytical_presolve", False)),
            differentiable_bridge=bool(payload.get("differentiable_bridge", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of backend settings.

        Returns:
            Dictionary with the backend enum serialized by value.
        """
        payload = asdict(self)
        payload["backend"] = self.backend.value
        return payload


def recommend_backend(
    num_params: int,
    residual_cfg: ResidualWeightConfig | None,
    param_entries: list[SysIdParameterEntry] | None = None,
    *,
    differentiable_bridge: bool = False,
) -> OptimizerBackend:
    """Pick CMA-ES for contact-heavy or high-dimensional problems (SRD REQ-7).

    When the active bridge exposes autograd rollouts, prefer gradient descent for
    everything a gradient can handle (one backward pass per iteration, independent
    of parameter count); command-delay and contact-force problems stay on CMA-ES.

    Args:
        num_params: Number of optimized parameter columns.
        residual_cfg: Enabled residual channels and their weights.
        param_entries: Parameter metadata used to detect delay rows.
        differentiable_bridge: Whether the active rollout bridge supports autograd.

    Returns:
        Recommended concrete optimizer backend.
    """
    cfg = residual_cfg or ResidualWeightConfig()
    if differentiable_bridge and not _has_command_delay_parameter(param_entries) and cfg.contact_force_weight <= 0.0:
        return OptimizerBackend.GRADIENT_DESCENT
    if _has_command_delay_parameter(param_entries):
        return OptimizerBackend.CMA_ES
    if cfg.contact_force_weight > 0.0:
        return OptimizerBackend.CMA_ES
    if num_params > HIGH_DIMENSION_THRESHOLD:
        return OptimizerBackend.CMA_ES
    return OptimizerBackend.LEVENBERG_MARQUARDT


def resolve_backend(
    backend_config: OptimizerBackendConfig,
    num_params: int,
    residual_cfg: ResidualWeightConfig | None,
    param_entries: list[SysIdParameterEntry] | None = None,
) -> OptimizerBackend:
    """Resolve auto backend settings to a concrete optimizer backend.

    Args:
        backend_config: Persisted backend choice and bridge capabilities.
        num_params: Number of optimized parameter columns.
        residual_cfg: Enabled residual channels and their weights.
        param_entries: Parameter metadata used for automatic selection.

    Returns:
        Explicitly configured backend, or the resolved automatic recommendation.
    """
    if backend_config.backend == OptimizerBackend.AUTO:
        return recommend_backend(
            num_params,
            residual_cfg,
            param_entries,
            differentiable_bridge=backend_config.differentiable_bridge,
        )
    return backend_config.backend


def _has_command_delay_parameter(
    param_entries: list[SysIdParameterEntry] | None,
) -> bool:
    return any(is_actuator_delay_parameter(entry.param_type) for entry in (param_entries or []))


def load_optimizer_backend_config(path: str | None = None) -> OptimizerBackendConfig:
    """Load portable optimizer settings from ``path`` or return defaults.

    Args:
        path: Optional JSON configuration path.

    Returns:
        Loaded configuration, or defaults when ``path`` is omitted.
    """
    if path is None:
        return OptimizerBackendConfig()
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"Optimizer configuration file does not exist: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Optimizer configuration must contain a JSON object: {file_path}")
    return OptimizerBackendConfig.from_dict(payload)
