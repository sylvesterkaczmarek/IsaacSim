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

"""Validate linear and angular DOF tensors with the OvPhysX engine."""

from __future__ import annotations

import os
import sys

import _physics_setup  # noqa: F401

_TENSORS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TENSORS_DIR not in sys.path:
    sys.path.append(_TENSORS_DIR)

from _legacy_runner import (  # noqa: E402
    cpu_device,
    gpu_device,
    gpu_only,
    run_scenario,
)
from common.linear_angular_dof import (  # noqa: E402
    AngularDofForcesCommon,
    AngularDofPositionsCommon,
    AngularDofPositionTargetsCommon,
    AngularDofVelocitiesCommon,
    AngularDofVelocityTargetsCommon,
    LinearDofForcesCommon,
    LinearDofPositionsCommon,
    LinearDofPositionTargetsCommon,
    LinearDofVelocitiesCommon,
    LinearDofVelocityTargetsCommon,
)


class TestLinearDofPositions:
    """Round-trip per-environment rail-cart positions through articulation tensors."""

    def test_linear_dof_positions_ovphysx_cc(self) -> None:
        """Round-trip distinct rail-cart positions with CPU simulation and CPU tensors."""
        run_scenario(self, LinearDofPositionsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_linear_dof_positions_ovphysx_gg(self) -> None:
        """Round-trip distinct rail-cart positions with GPU simulation and GPU tensors."""
        run_scenario(self, LinearDofPositionsCommon, "ovphysx", gpu_device())


class TestLinearDofVelocities:
    """Round-trip per-environment rail-cart velocities through articulation tensors."""

    def test_linear_dof_velocities_ovphysx_cc(self) -> None:
        """Round-trip distinct rail-cart velocities with CPU simulation and CPU tensors."""
        run_scenario(self, LinearDofVelocitiesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_linear_dof_velocities_ovphysx_gg(self) -> None:
        """Round-trip distinct rail-cart velocities with GPU simulation and GPU tensors."""
        run_scenario(self, LinearDofVelocitiesCommon, "ovphysx", gpu_device())


class TestAngularDofPositions:
    """Round-trip per-environment pole angles through articulation tensors."""

    def test_angular_dof_positions_ovphysx_cc(self) -> None:
        """Round-trip distinct pole angles with CPU simulation and CPU tensors."""
        run_scenario(self, AngularDofPositionsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_angular_dof_positions_ovphysx_gg(self) -> None:
        """Round-trip distinct pole angles with GPU simulation and GPU tensors."""
        run_scenario(self, AngularDofPositionsCommon, "ovphysx", gpu_device())


class TestAngularDofVelocities:
    """Round-trip per-environment pole angular velocities through tensors."""

    def test_angular_dof_velocities_ovphysx_cc(self) -> None:
        """Round-trip distinct pole angular velocities with CPU simulation and CPU tensors."""
        run_scenario(self, AngularDofVelocitiesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_angular_dof_velocities_ovphysx_gg(self) -> None:
        """Round-trip distinct pole angular velocities with GPU simulation and GPU tensors."""
        run_scenario(self, AngularDofVelocitiesCommon, "ovphysx", gpu_device())


class TestLinearDofForces:
    """Drive displaced rail carts to rest with tensor-computed actuation forces."""

    def test_linear_dof_forces_ovphysx_cc(self) -> None:
        """Drive displaced carts to zero state with CPU simulation and CPU tensors."""
        run_scenario(self, LinearDofForcesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_linear_dof_forces_ovphysx_gg(self) -> None:
        """Drive displaced carts to zero state with GPU simulation and GPU tensors."""
        run_scenario(self, LinearDofForcesCommon, "ovphysx", gpu_device())


class TestLinearDofPositionTargets:
    """Check rail-cart drives converge to per-environment position targets."""

    def test_linear_dof_position_targets_ovphysx_cc(self) -> None:
        """Check distinct cart position targets with CPU simulation and CPU tensors."""
        run_scenario(self, LinearDofPositionTargetsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_linear_dof_position_targets_ovphysx_gg(self) -> None:
        """Check distinct cart position targets with GPU simulation and GPU tensors."""
        run_scenario(self, LinearDofPositionTargetsCommon, "ovphysx", gpu_device())


class TestLinearDofVelocityTargets:
    """Check rail-cart drives converge to per-environment velocity targets."""

    def test_linear_dof_velocity_targets_ovphysx_cc(self) -> None:
        """Check distinct cart velocity targets with CPU simulation and CPU tensors."""
        run_scenario(self, LinearDofVelocityTargetsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_linear_dof_velocity_targets_ovphysx_gg(self) -> None:
        """Check distinct cart velocity targets with GPU simulation and GPU tensors."""
        run_scenario(self, LinearDofVelocityTargetsCommon, "ovphysx", gpu_device())


class TestAngularDofForces:
    """Compare pole responses with alternating positive and zero actuation torque."""

    def test_angular_dof_forces_ovphysx_cc(self) -> None:
        """Check torqued and unforced poles with CPU simulation and CPU tensors."""
        run_scenario(self, AngularDofForcesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_angular_dof_forces_ovphysx_gg(self) -> None:
        """Check torqued and unforced poles with GPU simulation and GPU tensors."""
        run_scenario(self, AngularDofForcesCommon, "ovphysx", gpu_device())


class TestAngularDofPositionTargets:
    """Check pole drives converge to per-environment angular position targets."""

    def test_angular_dof_position_targets_ovphysx_cc(self) -> None:
        """Check distinct pole position targets with CPU simulation and CPU tensors."""
        run_scenario(self, AngularDofPositionTargetsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_angular_dof_position_targets_ovphysx_gg(self) -> None:
        """Check distinct pole position targets with GPU simulation and GPU tensors."""
        run_scenario(self, AngularDofPositionTargetsCommon, "ovphysx", gpu_device())


class TestAngularDofVelocityTargets:
    """Check pole drives converge to per-environment angular velocity targets."""

    def test_angular_dof_velocity_targets_ovphysx_cc(self) -> None:
        """Check distinct pole velocity targets with CPU simulation and CPU tensors."""
        run_scenario(self, AngularDofVelocityTargetsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_angular_dof_velocity_targets_ovphysx_gg(self) -> None:
        """Check distinct pole velocity targets with GPU simulation and GPU tensors."""
        run_scenario(self, AngularDofVelocityTargetsCommon, "ovphysx", gpu_device())
