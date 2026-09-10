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

"""Rollout outputs from simulation bridges."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SysIdRolloutResult:
    """Batched rollout signals aligned with telemetry time indices."""

    positions: torch.Tensor
    velocities: torch.Tensor
    torques: torch.Tensor | None = None
    end_effector_poses: torch.Tensor | None = None
    contact_forces: torch.Tensor | None = None

    @property
    def num_envs(self) -> int:
        """Return the number of simulated environments."""
        return int(self.positions.shape[0])

    @property
    def num_steps(self) -> int:
        """Return the number of time samples in each rollout."""
        return int(self.positions.shape[1])
