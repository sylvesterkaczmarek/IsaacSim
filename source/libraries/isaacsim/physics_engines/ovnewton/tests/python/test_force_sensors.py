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

"""Validate force-sensor tensors with the Newton engine.

Force/torque sensors are skipped because Newton does not provide joint-force reporting.
Incoming-joint-force cases are skipped because the fixture cannot author the
required external forces through a Newton API. The joint-friction scenario
runs on each available device.
"""

from __future__ import annotations

import os
import sys

import _physics_setup  # noqa: F401
import pytest

_TENSORS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TENSORS_DIR not in sys.path:
    sys.path.append(_TENSORS_DIR)

from _legacy_runner import (  # noqa: E402
    cpu_device,
    gpu_device,
    gpu_only,
    run_scenario,
)
from common.force_sensors import (  # noqa: E402
    ForceTorqueSensorXCommon,
    ForceTorqueSensorYCommon,
    ForceTorqueSensorZCommon,
    JointFrictionCommon,
)

_GAP_JOINT_FORCE_REPORTING = (
    "Newton joint-force reporting (link-incoming-joint-force / "
    "dof-projected-joint-forces) is deferred pending a design decision: computing them "
    "requires requesting State.body_parent_f at model-build time, an always-on per-step "
    "cost on every solver (a full RNE post-constraint pass under MuJoCo) whether or not "
    "the tensor is read. Unconditional vs. opt-in gating is undecided; the ops stay "
    "unregistered until then."
)


@pytest.mark.skip(reason=_GAP_JOINT_FORCE_REPORTING)
class TestForceTorqueSensorY:
    """Skip Y-axis force/torque sensing because Newton does not report joint forces."""

    def test_force_torque_sensor_y_newton_cc(self) -> None:
        """Verify Y-axis force/torque sensing on the Newton CPU pipeline."""
        run_scenario(self, ForceTorqueSensorYCommon, "newton", cpu_device())

    @gpu_only
    def test_force_torque_sensor_y_newton_gg(self) -> None:
        """Verify Y-axis force/torque sensing on the Newton GPU pipeline."""
        run_scenario(self, ForceTorqueSensorYCommon, "newton", gpu_device())


@pytest.mark.skip(reason=_GAP_JOINT_FORCE_REPORTING)
class TestForceTorqueSensorX:
    """Skip X-axis force/torque sensing because Newton does not report joint forces."""

    def test_force_torque_sensor_x_newton_cc(self) -> None:
        """Verify X-axis force/torque sensing on the Newton CPU pipeline."""
        run_scenario(self, ForceTorqueSensorXCommon, "newton", cpu_device())

    @gpu_only
    def test_force_torque_sensor_x_newton_gg(self) -> None:
        """Verify X-axis force/torque sensing on the Newton GPU pipeline."""
        run_scenario(self, ForceTorqueSensorXCommon, "newton", gpu_device())


@pytest.mark.skip(reason=_GAP_JOINT_FORCE_REPORTING)
class TestForceTorqueSensorZ:
    """Skip Z-axis force/torque sensing because Newton does not report joint forces."""

    def test_force_torque_sensor_z_newton_cc(self) -> None:
        """Verify Z-axis force/torque sensing on the Newton CPU pipeline."""
        run_scenario(self, ForceTorqueSensorZCommon, "newton", cpu_device())

    @gpu_only
    def test_force_torque_sensor_z_newton_gg(self) -> None:
        """Verify Z-axis force/torque sensing on the Newton GPU pipeline."""
        run_scenario(self, ForceTorqueSensorZCommon, "newton", gpu_device())


# `LinkIncomingJointForceCommon` requires engine-specific external-force
# authoring (PhysxForceAPI on ovphysx); Newton has no analogue today.
# Skipped at the class level until Newton wires its own equivalent.
@pytest.mark.skip(
    reason=(
        "LinkIncomingJointForce scenarios require engine-specific external-force "
        "authoring (PhysxForceAPI on ovphysx); no Newton analogue yet."
    )
)
class TestLinkIncomingJointForceChildY:
    """Skip Y-axis child-link force checks without Newton force authoring."""

    def test_link_incoming_joint_force_child_y_newton_cc(self) -> None:
        """Record the unimplemented Y-axis child-link force case for Newton."""

    @gpu_only
    def test_link_incoming_joint_force_child_y_newton_gg(self) -> None:
        """Record the unimplemented Y-axis child-link force case for Newton."""


@pytest.mark.skip(
    reason=(
        "LinkIncomingJointForce scenarios require engine-specific external-force "
        "authoring (PhysxForceAPI on ovphysx); no Newton analogue yet."
    )
)
class TestLinkIncomingJointForceFixedY:
    """Skip Y-axis fixed-link force checks without Newton force authoring."""

    def test_link_incoming_joint_force_fixed_y_newton_cc(self) -> None:
        """Record the unimplemented Y-axis fixed-link force case for Newton."""

    @gpu_only
    def test_link_incoming_joint_force_fixed_y_newton_gg(self) -> None:
        """Record the unimplemented Y-axis fixed-link force case for Newton."""


@pytest.mark.skip(reason="no Newton analogue for PhysxForceAPI external-force injection")
class TestLinkIncomingJointForceChildX:
    """Skip X-axis child-link force checks without Newton force authoring."""

    def test_link_incoming_joint_force_child_x_newton_cc(self) -> None:
        """Record the unimplemented X-axis child-link force case for Newton."""

    @gpu_only
    def test_link_incoming_joint_force_child_x_newton_gg(self) -> None:
        """Record the unimplemented X-axis child-link force case for Newton."""


@pytest.mark.skip(reason="no Newton analogue for PhysxForceAPI external-force injection")
class TestLinkIncomingJointForceFixedX:
    """Skip X-axis fixed-link force checks without Newton force authoring."""

    def test_link_incoming_joint_force_fixed_x_newton_cc(self) -> None:
        """Record the unimplemented X-axis fixed-link force case for Newton."""

    @gpu_only
    def test_link_incoming_joint_force_fixed_x_newton_gg(self) -> None:
        """Record the unimplemented X-axis fixed-link force case for Newton."""


@pytest.mark.skip(reason="no Newton analogue for PhysxForceAPI external-force injection")
class TestLinkIncomingJointForceChildZ:
    """Skip Z-axis child-link force checks without Newton force authoring."""

    def test_link_incoming_joint_force_child_z_newton_cc(self) -> None:
        """Record the unimplemented Z-axis child-link force case for Newton."""

    @gpu_only
    def test_link_incoming_joint_force_child_z_newton_gg(self) -> None:
        """Record the unimplemented Z-axis child-link force case for Newton."""


@pytest.mark.skip(reason="no Newton analogue for PhysxForceAPI external-force injection")
class TestLinkIncomingJointForceFixedZ:
    """Skip Z-axis fixed-link force checks without Newton force authoring."""

    def test_link_incoming_joint_force_fixed_z_newton_cc(self) -> None:
        """Record the unimplemented Z-axis fixed-link force case for Newton."""

    @gpu_only
    def test_link_incoming_joint_force_fixed_z_newton_gg(self) -> None:
        """Record the unimplemented Z-axis fixed-link force case for Newton."""


class TestJointFriction:
    """Validate joint friction with Newton."""

    def test_joint_friction_newton_cc(self) -> None:
        """Verify joint friction on the Newton CPU pipeline."""
        run_scenario(self, JointFrictionCommon, "newton", cpu_device())

    @gpu_only
    def test_joint_friction_newton_gg(self) -> None:
        """Verify joint friction on the Newton GPU pipeline."""
        run_scenario(self, JointFrictionCommon, "newton", gpu_device())
