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

"""Residual weight configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .residual_engine import (
    DEFAULT_RESIDUAL_SPECS,
    ResidualSignalType,
    ResidualSpec,
    WeightedResidualEngine,
)


@dataclass
class ResidualWeightConfig:
    """Per-channel and optional per-sample residual weights."""

    position_weight: float = 1.0
    velocity_weight: float = 1.0
    torque_weight: float = 0.0
    end_effector_pose_weight: float = 0.0
    contact_force_weight: float = 0.0
    sample_weights: np.ndarray | None = None
    end_effector_link_index: int = -1

    def __post_init__(self) -> None:
        """Normalize and validate residual weights at the public model boundary."""
        for field_name in (
            "position_weight",
            "velocity_weight",
            "torque_weight",
            "end_effector_pose_weight",
            "contact_force_weight",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be a finite, non-negative number.")
            setattr(self, field_name, value)

        if self.sample_weights is not None:
            sample_weights = np.asarray(self.sample_weights, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights < 0.0):
                raise ValueError("sample_weights must contain only finite, non-negative numbers.")
            self.sample_weights = sample_weights

    def to_specs(self) -> tuple[ResidualSpec, ...]:
        """Convert channel weights into residual engine specs.

        Returns:
            Ordered residual specifications for every supported channel.
        """
        return (
            ResidualSpec(ResidualSignalType.POSITION, self.position_weight),
            ResidualSpec(ResidualSignalType.VELOCITY, self.velocity_weight),
            ResidualSpec(ResidualSignalType.TORQUE, self.torque_weight),
            ResidualSpec(ResidualSignalType.END_EFFECTOR_POSE, self.end_effector_pose_weight),
            ResidualSpec(ResidualSignalType.CONTACT_FORCE, self.contact_force_weight),
        )

    def build_engine(self) -> WeightedResidualEngine:
        """Build a residual engine for this weight configuration.

        Returns:
            Residual engine configured with these channel weights.
        """
        return WeightedResidualEngine(self.to_specs())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResidualWeightConfig:
        """Create residual weight settings from a JSON/YAML payload.

        Args:
            payload: Serialized input payload.

        Returns:
            Validated residual settings with NumPy sample weights.
        """
        sample_weights = payload.get("sample_weights")
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        return cls(
            position_weight=float(payload.get("position_weight", 1.0)),
            velocity_weight=float(payload.get("velocity_weight", 1.0)),
            torque_weight=float(payload.get("torque_weight", 0.0)),
            end_effector_pose_weight=float(payload.get("end_effector_pose_weight", 0.0)),
            contact_force_weight=float(payload.get("contact_force_weight", 0.0)),
            sample_weights=sample_weights,
            end_effector_link_index=int(payload.get("end_effector_link_index", -1)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of residual weight settings.

        Returns:
            JSON-compatible channel and sample weight mapping.
        """
        payload: dict[str, Any] = {
            "position_weight": self.position_weight,
            "velocity_weight": self.velocity_weight,
            "torque_weight": self.torque_weight,
            "end_effector_pose_weight": self.end_effector_pose_weight,
            "contact_force_weight": self.contact_force_weight,
            "end_effector_link_index": self.end_effector_link_index,
        }
        if self.sample_weights is not None:
            payload["sample_weights"] = self.sample_weights.tolist()
        return payload


def load_residual_weight_config(path: str | None) -> ResidualWeightConfig:
    """Load residual weights from JSON (returns defaults when path is empty).

    Args:
        path: Optional residual-weight JSON path.

    Returns:
        Loaded and validated weights, or defaults when the path is empty.
    """
    if not path:
        return ResidualWeightConfig()
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"Residual weight config not found: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Residual weight config must be a JSON object.")
    return ResidualWeightConfig.from_dict(payload)


def default_residual_weight_config() -> ResidualWeightConfig:
    """Return the default residual channel weights.

    Returns:
        Position/velocity defaults shared with the residual engine.
    """
    return ResidualWeightConfig(
        position_weight=DEFAULT_RESIDUAL_SPECS[0].weight,
        velocity_weight=DEFAULT_RESIDUAL_SPECS[1].weight,
    )
