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

"""Validate linear and angular DOF tensors with the Newton engine."""

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
    """Validate linear DOF positions with Newton."""

    def test_linear_dof_positions_newton_cc(self) -> None:
        """Verify linear DOF positions on the Newton CPU pipeline."""
        run_scenario(self, LinearDofPositionsCommon, "newton", cpu_device())

    @gpu_only
    def test_linear_dof_positions_newton_gg(self) -> None:
        """Verify linear DOF positions on the Newton GPU pipeline."""
        run_scenario(self, LinearDofPositionsCommon, "newton", gpu_device())


class TestLinearDofVelocities:
    """Validate linear DOF velocities with Newton."""

    def test_linear_dof_velocities_newton_cc(self) -> None:
        """Verify linear DOF velocities on the Newton CPU pipeline."""
        run_scenario(self, LinearDofVelocitiesCommon, "newton", cpu_device())

    @gpu_only
    def test_linear_dof_velocities_newton_gg(self) -> None:
        """Verify linear DOF velocities on the Newton GPU pipeline."""
        run_scenario(self, LinearDofVelocitiesCommon, "newton", gpu_device())


class TestAngularDofPositions:
    """Validate angular DOF positions with Newton."""

    def test_angular_dof_positions_newton_cc(self) -> None:
        """Verify angular DOF positions on the Newton CPU pipeline."""
        run_scenario(self, AngularDofPositionsCommon, "newton", cpu_device())

    @gpu_only
    def test_angular_dof_positions_newton_gg(self) -> None:
        """Verify angular DOF positions on the Newton GPU pipeline."""
        run_scenario(self, AngularDofPositionsCommon, "newton", gpu_device())


class TestAngularDofVelocities:
    """Validate angular DOF velocities with Newton."""

    def test_angular_dof_velocities_newton_cc(self) -> None:
        """Verify angular DOF velocities on the Newton CPU pipeline."""
        run_scenario(self, AngularDofVelocitiesCommon, "newton", cpu_device())

    @gpu_only
    def test_angular_dof_velocities_newton_gg(self) -> None:
        """Verify angular DOF velocities on the Newton GPU pipeline."""
        run_scenario(self, AngularDofVelocitiesCommon, "newton", gpu_device())


class TestLinearDofForces:
    """Validate linear DOF forces with Newton."""

    def test_linear_dof_forces_newton_cc(self) -> None:
        """Verify linear DOF forces on the Newton CPU pipeline."""
        run_scenario(self, LinearDofForcesCommon, "newton", cpu_device())

    @gpu_only
    def test_linear_dof_forces_newton_gg(self) -> None:
        """Verify linear DOF forces on the Newton GPU pipeline."""
        run_scenario(self, LinearDofForcesCommon, "newton", gpu_device())


class TestLinearDofPositionTargets:
    """Validate linear DOF position targets with Newton."""

    def test_linear_dof_position_targets_newton_cc(self) -> None:
        """Verify linear DOF position targets on the Newton CPU pipeline."""
        run_scenario(self, LinearDofPositionTargetsCommon, "newton", cpu_device())

    @gpu_only
    def test_linear_dof_position_targets_newton_gg(self) -> None:
        """Verify linear DOF position targets on the Newton GPU pipeline."""
        run_scenario(self, LinearDofPositionTargetsCommon, "newton", gpu_device())


class TestLinearDofVelocityTargets:
    """Validate linear DOF velocity targets with Newton."""

    def test_linear_dof_velocity_targets_newton_cc(self) -> None:
        """Verify linear DOF velocity targets on the Newton CPU pipeline."""
        run_scenario(self, LinearDofVelocityTargetsCommon, "newton", cpu_device())

    @gpu_only
    def test_linear_dof_velocity_targets_newton_gg(self) -> None:
        """Verify linear DOF velocity targets on the Newton GPU pipeline."""
        run_scenario(self, LinearDofVelocityTargetsCommon, "newton", gpu_device())


class TestAngularDofForces:
    """Validate angular DOF forces with Newton."""

    def test_angular_dof_forces_newton_cc(self) -> None:
        """Verify angular DOF forces on the Newton CPU pipeline."""
        run_scenario(self, AngularDofForcesCommon, "newton", cpu_device())

    @gpu_only
    def test_angular_dof_forces_newton_gg(self) -> None:
        """Verify angular DOF forces on the Newton GPU pipeline."""
        run_scenario(self, AngularDofForcesCommon, "newton", gpu_device())


class TestAngularDofPositionTargets:
    """Validate angular DOF position targets with Newton."""

    def test_angular_dof_position_targets_newton_cc(self) -> None:
        """Verify angular DOF position targets on the Newton CPU pipeline."""
        run_scenario(self, AngularDofPositionTargetsCommon, "newton", cpu_device())

    @gpu_only
    def test_angular_dof_position_targets_newton_gg(self) -> None:
        """Verify angular DOF position targets on the Newton GPU pipeline."""
        run_scenario(self, AngularDofPositionTargetsCommon, "newton", gpu_device())


class TestAngularDofVelocityTargets:
    """Validate angular DOF velocity targets with Newton."""

    def test_angular_dof_velocity_targets_newton_cc(self) -> None:
        """Verify angular DOF velocity targets on the Newton CPU pipeline."""
        run_scenario(self, AngularDofVelocityTargetsCommon, "newton", cpu_device())

    @gpu_only
    def test_angular_dof_velocity_targets_newton_gg(self) -> None:
        """Verify angular DOF velocity targets on the Newton GPU pipeline."""
        run_scenario(self, AngularDofVelocityTargetsCommon, "newton", gpu_device())
