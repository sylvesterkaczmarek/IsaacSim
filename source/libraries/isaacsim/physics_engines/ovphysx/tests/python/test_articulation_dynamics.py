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

"""Validate articulation dynamics tensors with the OvPhysX engine.

The engine-neutral scenarios use standard USD Physics authoring and therefore
require no additional PhysX schemas.
"""

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
from common.articulation_dynamics import (  # noqa: E402
    ArticulationRootTransformsCommon,
    ArticulationRootVelocitiesCommon,
    CoriolisCentrifugalCommon,
    DofDriveTypeCommon,
    DofVelocityThroughRootVelocitySetCommon,
    GravityCompensationCommon,
    JacobiansCommon,
    JointActuationForcesCommon,
    JointPerformanceEnvelopeCommon,
    LinkAccelerationsCommon,
    MassMatricesCommon,
    RootVelocityThroughDofSetCommon,
)


class TestJacobians:
    """Validate that Ant Jacobians are batched, finite, and nonzero."""

    def test_jacobians_ovphysx_cc(self) -> None:
        """Check finite, nonzero Ant Jacobians with CPU simulation and CPU tensors."""
        run_scenario(self, JacobiansCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_jacobians_ovphysx_gg(self) -> None:
        """Check finite, nonzero Ant Jacobians with GPU simulation and GPU tensors."""
        run_scenario(self, JacobiansCommon, "ovphysx", gpu_device())


class TestRootVelocityThroughDofSet:
    """Verify joint state writes preserve a previously assigned root velocity."""

    def test_root_velocity_through_dof_set_ovphysx_cc(self) -> None:
        """Check root-velocity preservation with CPU simulation and CPU tensors."""
        run_scenario(self, RootVelocityThroughDofSetCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_root_velocity_through_dof_set_ovphysx_gg(self) -> None:
        """Check root-velocity preservation with GPU simulation and GPU tensors."""
        run_scenario(self, RootVelocityThroughDofSetCommon, "ovphysx", gpu_device())


class TestDofVelocityThroughRootVelocitySet:
    """Verify a root-velocity write preserves assigned joint velocities."""

    def test_dof_velocity_through_root_velocity_set_ovphysx_cc(self) -> None:
        """Check joint-velocity preservation with CPU simulation and CPU tensors."""
        run_scenario(self, DofVelocityThroughRootVelocitySetCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_dof_velocity_through_root_velocity_set_ovphysx_gg(self) -> None:
        """Check joint-velocity preservation with GPU simulation and GPU tensors."""
        run_scenario(self, DofVelocityThroughRootVelocitySetCommon, "ovphysx", gpu_device())


class TestMassMatrices:
    """Validate Ant mass-matrix batching, finiteness, symmetry, and positive diagonals."""

    def test_mass_matrices_ovphysx_cc(self) -> None:
        """Check Ant mass-matrix invariants with CPU simulation and CPU tensors."""
        run_scenario(self, MassMatricesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_mass_matrices_ovphysx_gg(self) -> None:
        """Check Ant mass-matrix invariants with GPU simulation and GPU tensors."""
        run_scenario(self, MassMatricesCommon, "ovphysx", gpu_device())


class TestCoriolisCentrifugal:
    """Validate the batch dimension of Ant Coriolis and centrifugal forces."""

    def test_coriolis_centrifugal_ovphysx_cc(self) -> None:
        """Check compensation-force batching with CPU simulation and CPU tensors."""
        run_scenario(self, CoriolisCentrifugalCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_coriolis_centrifugal_ovphysx_gg(self) -> None:
        """Check compensation-force batching with GPU simulation and GPU tensors."""
        run_scenario(self, CoriolisCentrifugalCommon, "ovphysx", gpu_device())


class TestGravityCompensation:
    """Validate the batch dimension of Ant gravity-compensation forces."""

    def test_gravity_compensation_ovphysx_cc(self) -> None:
        """Check gravity-compensation batching with CPU simulation and CPU tensors."""
        run_scenario(self, GravityCompensationCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_gravity_compensation_ovphysx_gg(self) -> None:
        """Check gravity-compensation batching with GPU simulation and GPU tensors."""
        run_scenario(self, GravityCompensationCommon, "ovphysx", gpu_device())


class TestArticulationRootTransforms:
    """Verify root-pose writes propagate matching offsets to every Ant link."""

    def test_articulation_root_transforms_ovphysx_cc(self) -> None:
        """Check root and link pose propagation with CPU simulation and CPU tensors."""
        run_scenario(self, ArticulationRootTransformsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_root_transforms_ovphysx_gg(self) -> None:
        """Check root and link pose propagation with GPU simulation and GPU tensors."""
        run_scenario(self, ArticulationRootTransformsCommon, "ovphysx", gpu_device())


class TestArticulationRootVelocities:
    """Verify root-velocity writes propagate matching offsets to every Ant link."""

    def test_articulation_root_velocities_ovphysx_cc(self) -> None:
        """Check root and link velocity propagation with CPU simulation and CPU tensors."""
        run_scenario(self, ArticulationRootVelocitiesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_root_velocities_ovphysx_gg(self) -> None:
        """Check root and link velocity propagation with GPU simulation and GPU tensors."""
        run_scenario(self, ArticulationRootVelocitiesCommon, "ovphysx", gpu_device())


class TestLinkAccelerations:
    """Compare pendulum link angular acceleration with finite-difference joint acceleration."""

    def test_link_accelerations_ovphysx_cc(self) -> None:
        """Compare link and joint accelerations with CPU simulation and CPU tensors."""
        run_scenario(self, LinkAccelerationsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_link_accelerations_ovphysx_gg(self) -> None:
        """Compare link and joint accelerations with GPU simulation and GPU tensors."""
        run_scenario(self, LinkAccelerationsCommon, "ovphysx", gpu_device())


class TestJointActuationForces:
    """Verify projected CartPole joint forces track uniform applied actuation."""

    def test_joint_actuation_forces_ovphysx_cc(self) -> None:
        """Compare projected and applied forces with CPU simulation and CPU tensors."""
        run_scenario(self, JointActuationForcesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_joint_actuation_forces_ovphysx_gg(self) -> None:
        """Compare projected and applied forces with GPU simulation and GPU tensors."""
        run_scenario(self, JointActuationForcesCommon, "ovphysx", gpu_device())


class TestArticulationDofDriveType:
    """Validate the force and acceleration drive types the engine reports."""

    def test_articulation_dof_drive_type_ovphysx_cc(self) -> None:
        """Check reported drive types with CPU simulation and CPU tensors."""
        run_scenario(self, DofDriveTypeCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_dof_drive_type_ovphysx_gg(self) -> None:
        """Check reported drive types with GPU simulation and GPU tensors."""
        run_scenario(self, DofDriveTypeCommon, "ovphysx", gpu_device())


class TestJointPerformanceEnvelope:
    """Validate writable drive-model properties on a Franka articulation.

    The scenario requires DOF drive-model updates through the tensor API and
    treats a read-only or unavailable binding as a test failure.
    """

    def test_joint_performance_envelope_ovphysx_cc(self) -> None:
        """Write and read Franka drive-model data with CPU simulation and CPU tensors."""
        run_scenario(self, JointPerformanceEnvelopeCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_joint_performance_envelope_ovphysx_gg(self) -> None:
        """Write and read Franka drive-model data with GPU simulation and GPU tensors."""
        run_scenario(self, JointPerformanceEnvelopeCommon, "ovphysx", gpu_device())


# `joint_force_dof_projection` and `link_incoming_joint_force` placeholders
# previously lived here as `TestPortShells` (4 skipped methods) because the
# legacy `tests.py` mixed them into `testArticulationDynamics.py`. The real
# scenarios are now live under `TestForceProjection*` in
# `physx/test_force_projection.py` and `TestLinkIncomingJointForce*` in
# `physx/test_force_sensors.py` — the dynamics-file placeholders only added
# misleading SKIP rows to the test matrix without exercising anything new.
