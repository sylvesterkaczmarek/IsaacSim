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

"""Internal bridge protocol between SysId optimizers and simulation backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from .parameter_types import SysIdParameterEntry
    from .rollout_result import SysIdRolloutResult
    from .trajectory_csv import TrajectoryDataset


class SysIdEnvironmentBridgeError(RuntimeError):
    """Raised when the environment bridge is not configured or misused."""


@runtime_checkable
class SysIdEnvironmentBridge(Protocol):
    """Internal interface for vectorized simulation used during SysId rollouts.

    Optimizer backends call this protocol after the extension has wired an active
    simulation bridge. Bridge implementations run a full trajectory under the
    given parameter vectors and return simulated signals aligned with the telemetry
    time indices.

    Parallel layout for M selected parameters:

    - ``num_envs = M + 1`` (or sequential reuse of env 0)
    - Env 0: nominal parameter vector
    - Env i (1..M): nominal with parameter i perturbed by +epsilon

    """

    device: torch.device

    @property
    def supports_arbitrary_candidate_batch(self) -> bool:
        """Whether the bridge can evaluate arbitrary candidate rows in parallel."""
        ...

    def set_trajectory(self, trajectory: TrajectoryDataset) -> None:
        """Store the active trajectory for resets and command playback.

        Args:
            trajectory: Telemetry trajectory used by the operation.
        """
        ...

    async def run_rollout_async(
        self,
        theta_per_env: torch.Tensor,
        param_entries: list[SysIdParameterEntry],
        commands: torch.Tensor,
        num_steps: int,
    ) -> SysIdRolloutResult:
        """Run a full trajectory rollout.

        Row convention (all bridges): output row ``k`` is the simulated state at
        measured sample ``k``. Row 0 is the trajectory's initial state (measured
        row 0), and stepping the physics under command row ``k`` produces row
        ``k + 1`` — so the last command row never drives a step. Torque row ``k``
        is the actuation applied AT sample ``k`` (computed from state row ``k``
        and command row ``k``), matching the pre-step measured-torque convention.
        Residuals therefore compare sim row ``k`` against measured row ``k``
        with no time skew.

        Returns:
            Structured simulated signals shaped ``(num_envs, num_steps, *)``
            on ``self.device``.

        Args:
            theta_per_env: Candidate parameter rows with one row per environment.
            param_entries: Metadata describing each theta column.
            commands: Command trajectory shared by the candidate environments.
            num_steps: Number of simulation steps to execute.
        """
        ...

    def cleanup(self, remove_clones: bool = True) -> None:
        """Release subscriptions and cached simulation state.

        Args:
            remove_clones: Whether to remove generated simulation environments.
        """
        ...


class NullSysIdEnvironmentBridge:
    """Sentinel bridge used when no simulation backend is configured."""

    device = torch.device("cpu")
    supports_arbitrary_candidate_batch = False

    _MESSAGE = (
        "SysId environment bridge is not configured. Select a supported simulation backend "
        "and attach it to the optimizer before starting a rollout."
    )

    def set_trajectory(self, trajectory: TrajectoryDataset) -> None:
        """Reject trajectory assignment because no simulation bridge is configured.

        Args:
            trajectory: Telemetry trajectory that cannot be assigned.
        """
        raise SysIdEnvironmentBridgeError(self._MESSAGE)

    def cleanup(self, remove_clones: bool = True) -> None:
        """Leave the unconfigured bridge unchanged.

        Args:
            remove_clones: Ignored because no simulation environments exist.
        """
        _ = remove_clones

    async def run_rollout_async(
        self,
        theta_per_env: torch.Tensor,
        param_entries: list[SysIdParameterEntry],
        commands: torch.Tensor,
        num_steps: int,
    ) -> SysIdRolloutResult:
        """Reject rollout execution because no simulation bridge is configured.

        Args:
            theta_per_env: Candidate parameter rows.
            param_entries: Metadata describing each parameter column.
            commands: Command trajectory.
            num_steps: Requested simulation step count.

        Returns:
            This method does not return.
        """
        raise SysIdEnvironmentBridgeError(self._MESSAGE)
