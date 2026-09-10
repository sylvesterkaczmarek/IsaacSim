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

"""Validate articulation joint and body ordering with the OvPhysX engine.

These tests extend the engine-neutral scenarios with
``PhysxSchema.JointStateAPI`` values that seed each cart and pole joint before
the engine parses the stage.
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

# `common.X` works under `--import-mode=importlib` since
# `tests/python/` is on sys.path (via the parent conftest.py) and
# `tests/python/tensors/__init__.py` exists.
from common.articulation_joint_body_order import (  # noqa: E402
    JointBodyOrderDofForceCommon,
    JointBodyOrderLimitsCommon,
    JointBodyOrderLinkForceCommon,
    JointBodyOrderPositionCommon,
    JointBodyOrderPositionTargetCommon,
    JointBodyOrderVelocityCommon,
    JointBodyOrderVelocityTargetCommon,
)
from physx_usd_schemas import PhysxSchema  # noqa: E402

# ---------------------------------------------------------------------------
# ovphysx engine-specific extension.
# ---------------------------------------------------------------------------


class _PhysxJointStateMixin:
    """Apply `PhysxSchema.JointStateAPI` to every cart + pole joint."""

    def _apply_engine_specifics(self) -> None:
        super()._apply_engine_specifics()
        for i in range(self.num_envs):
            for env in (1, 2):
                state = PhysxSchema.JointStateAPI.Apply(
                    self.stage.GetPrimAtPath(f"/envs/env{i}/cartpole{env}/cartJoint"),
                    "linear",
                )
                state.CreatePositionAttr().Set(1.0)
                state.CreateVelocityAttr().Set(2.0)

                state = PhysxSchema.JointStateAPI.Apply(
                    self.stage.GetPrimAtPath(f"/envs/env{i}/cartpole{env}/poleJoint"),
                    "angular",
                )
                state.CreatePositionAttr().Set(30.0)
                state.CreateVelocityAttr().Set(40.0)


class _PhysxJointStateAtRestMixin:
    """Initialize each cart-pole joint at rest.

    The force scenarios use zero position and velocity so their assertions
    measure only the authored drives and simulated forces.
    """

    def _apply_engine_specifics(self) -> None:
        super()._apply_engine_specifics()
        for i in range(self.num_envs):
            for env in (1, 2):
                state = PhysxSchema.JointStateAPI.Apply(
                    self.stage.GetPrimAtPath(f"/envs/env{i}/cartpole{env}/cartJoint"),
                    "linear",
                )
                state.CreatePositionAttr().Set(0.0)
                state.CreateVelocityAttr().Set(0.0)
                state = PhysxSchema.JointStateAPI.Apply(
                    self.stage.GetPrimAtPath(f"/envs/env{i}/cartpole{env}/poleJoint"),
                    "angular",
                )
                state.CreatePositionAttr().Set(0.0)
                state.CreateVelocityAttr().Set(0.0)


class _LimitsScenario(_PhysxJointStateMixin, JointBodyOrderLimitsCommon):
    pass


class _PositionScenario(_PhysxJointStateMixin, JointBodyOrderPositionCommon):
    pass


class _VelocityScenario(_PhysxJointStateMixin, JointBodyOrderVelocityCommon):
    pass


class _PositionTargetScenario(_PhysxJointStateMixin, JointBodyOrderPositionTargetCommon):
    pass


class _VelocityTargetScenario(_PhysxJointStateMixin, JointBodyOrderVelocityTargetCommon):
    pass


class _DofForceScenario(_PhysxJointStateAtRestMixin, JointBodyOrderDofForceCommon):
    pass


class _LinkForceScenario(_PhysxJointStateAtRestMixin, JointBodyOrderLinkForceCommon):
    pass


# ---------------------------------------------------------------------------
# Test classes.
# ---------------------------------------------------------------------------


class TestArticulationJointBodyOrderLimits:
    """Compare joint limits across CartPoles with reversed joint body relationships."""

    def test_articulation_joint_body_order_dof_limits_ovphysx_cc(self) -> None:
        """Compare reversed-order joint limits with CPU simulation and CPU tensors."""
        run_scenario(self, _LimitsScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_joint_body_order_dof_limits_ovphysx_gg(self) -> None:
        """Compare reversed-order joint limits with GPU simulation and GPU tensors."""
        run_scenario(self, _LimitsScenario, "ovphysx", gpu_device())


class TestArticulationJointBodyOrderPosition:
    """Compare seeded joint positions across reversed body relationships."""

    def test_articulation_joint_body_order_position_ovphysx_cc(self) -> None:
        """Compare reversed-order joint positions with CPU simulation and CPU tensors."""
        run_scenario(self, _PositionScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_joint_body_order_position_ovphysx_gg(self) -> None:
        """Compare reversed-order joint positions with GPU simulation and GPU tensors."""
        run_scenario(self, _PositionScenario, "ovphysx", gpu_device())


class TestArticulationJointBodyOrderVelocity:
    """Compare seeded joint velocities across reversed body relationships."""

    def test_articulation_joint_body_order_velocity_ovphysx_cc(self) -> None:
        """Compare reversed-order joint velocities with CPU simulation and CPU tensors."""
        run_scenario(self, _VelocityScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_joint_body_order_velocity_ovphysx_gg(self) -> None:
        """Compare reversed-order joint velocities with GPU simulation and GPU tensors."""
        run_scenario(self, _VelocityScenario, "ovphysx", gpu_device())


class TestArticulationJointBodyOrderPositionTarget:
    """Compare position targets across reversed joint body relationships."""

    def test_articulation_joint_body_order_position_target_ovphysx_cc(self) -> None:
        """Compare reversed-order position targets with CPU simulation and CPU tensors."""
        run_scenario(self, _PositionTargetScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_joint_body_order_position_target_ovphysx_gg(self) -> None:
        """Compare reversed-order position targets with GPU simulation and GPU tensors."""
        run_scenario(self, _PositionTargetScenario, "ovphysx", gpu_device())


class TestArticulationJointBodyOrderVelocityTarget:
    """Compare velocity targets across reversed joint body relationships."""

    def test_articulation_joint_body_order_velocity_target_ovphysx_cc(self) -> None:
        """Compare reversed-order velocity targets with CPU simulation and CPU tensors."""
        run_scenario(self, _VelocityTargetScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_joint_body_order_velocity_target_ovphysx_gg(self) -> None:
        """Compare reversed-order velocity targets with GPU simulation and GPU tensors."""
        run_scenario(self, _VelocityTargetScenario, "ovphysx", gpu_device())


class TestArticulationJointBodyOrderDofForce:
    """Compare actuation and compensation forces across reversed body orderings."""

    def test_articulation_joint_body_order_dof_force_ovphysx_cc(self) -> None:
        """Compare reversed-order DOF forces with CPU simulation and CPU tensors."""
        run_scenario(self, _DofForceScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_joint_body_order_dof_force_ovphysx_gg(self) -> None:
        """Compare reversed-order DOF forces with GPU simulation and GPU tensors."""
        run_scenario(self, _DofForceScenario, "ovphysx", gpu_device())


class TestArticulationJointBodyOrderLinkForce:
    """Validate projected DOF and incoming-link forces after a body-order swap.

    The scenario applies sign-adjusted drive targets to equivalent cart-poles,
    then compares projected DOF forces and incoming-link force tensor shapes.
    """

    def test_articulation_joint_body_order_link_force_ovphysx_cc(self) -> None:
        """Compare projected and link-force data with CPU simulation and CPU tensors."""
        run_scenario(self, _LinkForceScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_joint_body_order_link_force_ovphysx_gg(self) -> None:
        """Compare projected and link-force data with GPU simulation and GPU tensors."""
        run_scenario(self, _LinkForceScenario, "ovphysx", gpu_device())
