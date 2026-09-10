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

"""Weighted residual assembly for rollout-based system identification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch


class ResidualSignalType(Enum):
    """Telemetry channels compared during optimization."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    END_EFFECTOR_POSE = "end_effector_pose"
    CONTACT_FORCE = "contact_force"


@dataclass(frozen=True)
class ResidualSpec:
    """One weighted residual channel."""

    signal: ResidualSignalType
    weight: float = 1.0


DEFAULT_RESIDUAL_SPECS: tuple[ResidualSpec, ...] = (
    ResidualSpec(ResidualSignalType.POSITION, weight=1.0),
    ResidualSpec(ResidualSignalType.VELOCITY, weight=1.0),
)


@dataclass(frozen=True)
class RolloutResidualInputs:
    """Measured and simulated trajectories for one rollout batch."""

    meas_pos: torch.Tensor
    meas_vel: torch.Tensor
    sim_pos: torch.Tensor
    sim_vel: torch.Tensor
    meas_torque: torch.Tensor | None = None
    sim_torque: torch.Tensor | None = None
    meas_ee_pose: torch.Tensor | None = None
    sim_ee_pose: torch.Tensor | None = None
    meas_contact: torch.Tensor | None = None
    sim_contact: torch.Tensor | None = None
    sample_weights: torch.Tensor | None = None


class WeightedResidualEngine:
    """Concatenate and weight residual vectors from rollout comparison data.

    Args:
        specs: Constructor value for ``specs``.
    """

    def __init__(self, specs: tuple[ResidualSpec, ...] = DEFAULT_RESIDUAL_SPECS) -> None:
        if not specs:
            raise ValueError("At least one ResidualSpec is required.")
        for spec in specs:
            weight = float(spec.weight)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"Residual weight for {spec.signal.value} must be a finite, non-negative number.")
        self._specs = specs

    @property
    def specs(self) -> tuple[ResidualSpec, ...]:
        """Return the residual specs used by this engine."""
        return self._specs

    @staticmethod
    def _broadcast_measured(meas: torch.Tensor, num_envs: int) -> torch.Tensor:
        if meas.ndim == 3:
            return meas
        return meas.unsqueeze(0).expand(num_envs, -1, -1)

    @staticmethod
    def _weighted_flatten(
        diff: torch.Tensor,
        sample_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Flatten ``(num_envs, T, D)`` diffs and apply optional per-sample weights.

        Args:
            diff: Batched channel differences with shape ``(B, T, D)``.
            sample_weights: Optional length-``T`` residual multipliers.

        Returns:
            Weighted residual rows flattened to shape ``(B, T*D)``.
        """
        num_envs, num_steps = diff.shape[0], diff.shape[1]
        if sample_weights is not None:
            flat_weights = sample_weights.reshape(-1)
            if flat_weights.numel() != num_steps:
                raise ValueError(f"sample_weights must contain exactly {num_steps} values, got {flat_weights.numel()}.")
            if bool(torch.any(~torch.isfinite(flat_weights) | (flat_weights < 0.0)).item()):
                raise ValueError("sample_weights must contain only finite, non-negative values.")
            weights = flat_weights.to(device=diff.device, dtype=diff.dtype).view(1, num_steps, 1)
            diff = diff * weights
        return diff.reshape(num_envs, -1)

    @staticmethod
    def _end_effector_pose_difference(sim_pose: torch.Tensor, meas_pose: torch.Tensor) -> torch.Tensor:
        """Return translation and sign-invariant rotation-vector errors.

        Seven-component poses use the documented ``xyz+xyzw`` convention.
        Nonstandard pose widths retain component-wise behavior for compatibility
        with application-defined feature vectors.

        Args:
            sim_pose: Simulated pose batch.
            meas_pose: Measured pose batch broadcast to the simulation shape.

        Returns:
            Translation plus rotation-vector errors for standard poses, or
            component-wise errors for nonstandard feature widths.
        """
        if sim_pose.shape[-1] != 7 or meas_pose.shape[-1] != 7:
            return sim_pose - meas_pose

        translation = sim_pose[..., :3] - meas_pose[..., :3]
        sim_quat = sim_pose[..., 3:7]
        meas_quat = meas_pose[..., 3:7]
        epsilon = torch.finfo(sim_pose.dtype).eps
        sim_norm = torch.linalg.vector_norm(sim_quat, dim=-1, keepdim=True)
        meas_norm = torch.linalg.vector_norm(meas_quat, dim=-1, keepdim=True)
        if bool(torch.any(~torch.isfinite(sim_norm) | (sim_norm <= epsilon)).item()):
            raise ValueError("Simulated end-effector pose contains a quaternion with zero or non-finite norm.")
        if bool(torch.any(~torch.isfinite(meas_norm) | (meas_norm <= epsilon)).item()):
            raise ValueError("Measured end-effector pose contains a quaternion with zero or non-finite norm.")
        sim_quat = sim_quat / sim_norm
        meas_quat = meas_quat / meas_norm

        dot = (sim_quat * meas_quat).sum(dim=-1, keepdim=True)
        meas_quat = torch.where(dot < 0.0, -meas_quat, meas_quat)
        sim_xyz, sim_w = sim_quat[..., :3], sim_quat[..., 3:4]
        meas_xyz, meas_w = meas_quat[..., :3], meas_quat[..., 3:4]
        relative_xyz = meas_w * sim_xyz - sim_w * meas_xyz - torch.linalg.cross(meas_xyz, sim_xyz, dim=-1)
        relative_w = (meas_w * sim_w + (meas_xyz * sim_xyz).sum(dim=-1, keepdim=True)).clamp_min(0.0)
        vector_norm = torch.linalg.vector_norm(relative_xyz, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(vector_norm, relative_w)
        scale = torch.where(
            vector_norm > epsilon,
            angle / vector_norm.clamp_min(epsilon),
            torch.full_like(angle, 2.0),
        )
        rotation_vector = relative_xyz * scale
        return torch.cat((translation, rotation_vector), dim=-1)

    def compute(self, inputs: RolloutResidualInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-env residual vectors and scalar costs.

        Args:
            inputs: Batch tensors with shapes ``(num_envs, num_steps, *)`` for simulated
                states and ``(num_steps, *)`` (or batched) for measured states.

        Returns:
            Tuple ``(residuals, cost_per_env)`` with shapes ``(num_envs, R)`` and
            ``(num_envs,)`` where cost is ``0.5 * ||r||^2`` per environment.
        """
        num_envs = inputs.sim_pos.shape[0]
        num_steps = inputs.sim_pos.shape[1]
        meas_pos_b = self._broadcast_measured(inputs.meas_pos, num_envs)
        meas_vel_b = self._broadcast_measured(inputs.meas_vel, num_envs)

        chunks: list[torch.Tensor] = []
        for spec in self._specs:
            weight = float(spec.weight)
            if weight == 0.0:
                continue
            sqrt_w = weight**0.5

            if spec.signal == ResidualSignalType.POSITION:
                diff = inputs.sim_pos - meas_pos_b
                err = self._weighted_flatten(diff, inputs.sample_weights) * sqrt_w
                chunks.append(err)
            elif spec.signal == ResidualSignalType.VELOCITY:
                diff = inputs.sim_vel - meas_vel_b
                err = self._weighted_flatten(diff, inputs.sample_weights) * sqrt_w
                chunks.append(err)
            elif spec.signal == ResidualSignalType.TORQUE:
                if inputs.meas_torque is None or inputs.sim_torque is None:
                    missing = "measured" if inputs.meas_torque is None else "simulated"
                    raise ValueError(f"Torque residual has nonzero weight but {missing} torque is unavailable.")
                meas_t = self._broadcast_measured(inputs.meas_torque, num_envs)
                diff = inputs.sim_torque[:, :num_steps] - meas_t[:, :num_steps]
                err = self._weighted_flatten(diff, inputs.sample_weights) * sqrt_w
                chunks.append(err)
            elif spec.signal == ResidualSignalType.END_EFFECTOR_POSE:
                if inputs.meas_ee_pose is None or inputs.sim_ee_pose is None:
                    missing = "measured" if inputs.meas_ee_pose is None else "simulated"
                    raise ValueError(
                        f"End-effector pose residual has nonzero weight but {missing} end-effector pose is unavailable."
                    )
                meas_ee = self._broadcast_measured(inputs.meas_ee_pose, num_envs)
                sim_ee = inputs.sim_ee_pose[:, :num_steps]
                diff = self._end_effector_pose_difference(sim_ee, meas_ee[:, :num_steps])
                err = self._weighted_flatten(diff, inputs.sample_weights) * sqrt_w
                chunks.append(err)
            elif spec.signal == ResidualSignalType.CONTACT_FORCE:
                if inputs.meas_contact is None or inputs.sim_contact is None:
                    missing = "measured" if inputs.meas_contact is None else "simulated"
                    raise ValueError(
                        f"Contact-force residual has nonzero weight but {missing} contact force is unavailable."
                    )
                meas_c = self._broadcast_measured(inputs.meas_contact, num_envs)
                diff = inputs.sim_contact[:, :num_steps] - meas_c[:, :num_steps]
                err = self._weighted_flatten(diff, inputs.sample_weights) * sqrt_w
                chunks.append(err)
            else:
                raise ValueError(f"Unknown residual signal: {spec.signal!r}")

        if not chunks:
            raise ValueError("All residual weights are zero or no comparable signals are available.")

        residuals = torch.cat(chunks, dim=1)
        cost_per_env = 0.5 * (residuals * residuals).sum(dim=1)
        return residuals, cost_per_env
