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

"""Validate nested rigid-body and articulation paths with OvPhysX.

The articulation scenarios add PhysX settings that disable sleeping and
self-collision. The tests cover explicit paths, recursive patterns, wildcard
patterns, alternation, and nested articulation roots.
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
from common.nested_rigid_bodies import (  # noqa: E402
    ArticulationViewAtNestedRootLinkCommon,
    ArticulationViewNestedLinksCommon,
    ArticulationViewRecursiveRootLinkCommon,
    RigidBodyViewExplicitNestedPathsCommon,
    RigidBodyViewNestedAlternationCommon,
    RigidBodyViewRecursiveNamedLinkCommon,
    RigidBodyViewRecursiveUnderRobotCommon,
    RigidBodyViewStandaloneExplicitPathsCommon,
    RigidBodyViewStandaloneRecursiveUnderClusterCommon,
    RigidBodyViewWildcardUnderRobotCommon,
)
from physx_usd_schemas import PhysxSchema  # noqa: E402

# ---------------------------------------------------------------------------
# ovphysx engine-specific extension.
# ---------------------------------------------------------------------------


class _PhysxArticulationApiMixin:
    """Configure each nested robot articulation for deterministic tests.

    The mixin disables sleeping and self-collision on every replicated robot
    root.
    """

    def _apply_engine_specifics(self) -> None:
        super()._apply_engine_specifics()
        for i in range(self.num_envs):
            prim = self.stage.GetPrimAtPath(f"/envs/env{i}/Robot")
            if prim:
                arti_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                arti_api.CreateSleepThresholdAttr(0.0)
                arti_api.CreateEnabledSelfCollisionsAttr().Set(False)


class _ArticulationViewNestedLinksScenario(_PhysxArticulationApiMixin, ArticulationViewNestedLinksCommon):
    pass


class _ArticulationViewAtNestedRootLinkScenario(_PhysxArticulationApiMixin, ArticulationViewAtNestedRootLinkCommon):
    pass


# ---------------------------------------------------------------------------
# Test classes.
# ---------------------------------------------------------------------------


class TestArticulationViewNestedLinks:
    """Resolve four-link, three-DOF articulations from their outer Robot prims."""

    def test_articulation_view_nested_links_ovphysx_cc(self) -> None:
        """Check topology and root paths with CPU simulation and CPU tensors."""
        run_scenario(self, _ArticulationViewNestedLinksScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_view_nested_links_ovphysx_gg(self) -> None:
        """Check topology and root paths with GPU simulation and GPU tensors."""
        run_scenario(self, _ArticulationViewNestedLinksScenario, "ovphysx", gpu_device())


class TestArticulationViewAtNestedRootLink:
    """Resolve four-link, three-DOF articulations from nested base-link prims."""

    def test_articulation_view_at_nested_root_link_ovphysx_cc(self) -> None:
        """Check base-link articulation resolution with CPU simulation and CPU tensors."""
        run_scenario(self, _ArticulationViewAtNestedRootLinkScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_view_at_nested_root_link_ovphysx_gg(self) -> None:
        """Check base-link articulation resolution with GPU simulation and GPU tensors."""
        run_scenario(self, _ArticulationViewAtNestedRootLinkScenario, "ovphysx", gpu_device())


class _ExplicitNestedScenario(_PhysxArticulationApiMixin, RigidBodyViewExplicitNestedPathsCommon):
    pass


class _RecursiveUnderRobotScenario(_PhysxArticulationApiMixin, RigidBodyViewRecursiveUnderRobotCommon):
    pass


class _RecursiveNamedLinkScenario(_PhysxArticulationApiMixin, RigidBodyViewRecursiveNamedLinkCommon):
    pass


class _ArticulationRecursiveScenario(_PhysxArticulationApiMixin, ArticulationViewRecursiveRootLinkCommon):
    pass


class _WildcardUnderRobotScenario(_PhysxArticulationApiMixin, RigidBodyViewWildcardUnderRobotCommon):
    pass


class _NestedAlternationScenario(_PhysxArticulationApiMixin, RigidBodyViewNestedAlternationCommon):
    pass


# Standalone-cluster variants don't need PhysxArticulationAPI.
_StandaloneExplicitScenario = RigidBodyViewStandaloneExplicitPathsCommon
_StandaloneRecursiveScenario = RigidBodyViewStandaloneRecursiveUnderClusterCommon


class TestRigidBodyViewExplicitNestedPaths:
    """Resolve one deepest articulation link from each explicit environment path."""

    def test_rigid_body_view_explicit_nested_paths_ovphysx_cc(self) -> None:
        """Resolve explicit ``link_2`` paths with CPU simulation and CPU tensors."""
        run_scenario(self, _ExplicitNestedScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_rigid_body_view_explicit_nested_paths_ovphysx_gg(self) -> None:
        """Resolve explicit ``link_2`` paths with GPU simulation and GPU tensors."""
        run_scenario(self, _ExplicitNestedScenario, "ovphysx", gpu_device())


class TestRigidBodyViewRecursiveUnderRobot:
    """Resolve all four rigid links below each Robot through recursive descent."""

    def test_rigid_body_view_recursive_under_robot_ovphysx_cc(self) -> None:
        """Resolve every Robot descendant with CPU simulation and CPU tensors."""
        run_scenario(self, _RecursiveUnderRobotScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_rigid_body_view_recursive_under_robot_ovphysx_gg(self) -> None:
        """Resolve every Robot descendant with GPU simulation and GPU tensors."""
        run_scenario(self, _RecursiveUnderRobotScenario, "ovphysx", gpu_device())


class TestRigidBodyViewRecursiveNamedLink:
    """Resolve one deepest ``link_2`` per Robot with a recursive-tail pattern."""

    def test_rigid_body_view_recursive_named_link_ovphysx_cc(self) -> None:
        """Resolve recursive ``link_2`` matches with CPU simulation and CPU tensors."""
        run_scenario(self, _RecursiveNamedLinkScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_rigid_body_view_recursive_named_link_ovphysx_gg(self) -> None:
        """Resolve recursive ``link_2`` matches with GPU simulation and GPU tensors."""
        run_scenario(self, _RecursiveNamedLinkScenario, "ovphysx", gpu_device())


class TestArticulationViewRecursiveRootLink:
    """Resolve nested articulation topology with a recursive base-link pattern."""

    def test_articulation_view_recursive_root_link_ovphysx_cc(self) -> None:
        """Resolve recursive base links with CPU simulation and CPU tensors."""
        run_scenario(self, _ArticulationRecursiveScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_view_recursive_root_link_ovphysx_gg(self) -> None:
        """Resolve recursive base links with GPU simulation and GPU tensors."""
        run_scenario(self, _ArticulationRecursiveScenario, "ovphysx", gpu_device())


class TestRigidBodyViewWildcardUnderRobot:
    """Ensure a one-level Robot wildcard selects only each base link."""

    def test_rigid_body_view_wildcard_under_robot_ovphysx_cc(self) -> None:
        """Check non-recursive base-link matching with CPU simulation and CPU tensors."""
        run_scenario(self, _WildcardUnderRobotScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_rigid_body_view_wildcard_under_robot_ovphysx_gg(self) -> None:
        """Check non-recursive base-link matching with GPU simulation and GPU tensors."""
        run_scenario(self, _WildcardUnderRobotScenario, "ovphysx", gpu_device())


class TestRigidBodyViewNestedAlternation:
    """Resolve each base link and first child through leaf-name alternation."""

    def test_rigid_body_view_nested_alternation_ovphysx_cc(self) -> None:
        """Resolve two alternated links per Robot with CPU simulation and CPU tensors."""
        run_scenario(self, _NestedAlternationScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_rigid_body_view_nested_alternation_ovphysx_gg(self) -> None:
        """Resolve two alternated links per Robot with GPU simulation and GPU tensors."""
        run_scenario(self, _NestedAlternationScenario, "ovphysx", gpu_device())


class TestRigidBodyViewStandaloneExplicitPaths:
    """Resolve every non-articulated cluster body from an explicit path list."""

    def test_rigid_body_view_standalone_explicit_paths_ovphysx_cc(self) -> None:
        """Resolve all explicit cluster bodies with CPU simulation and CPU tensors."""
        run_scenario(self, _StandaloneExplicitScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_rigid_body_view_standalone_explicit_paths_ovphysx_gg(self) -> None:
        """Resolve all explicit cluster bodies with GPU simulation and GPU tensors."""
        run_scenario(self, _StandaloneExplicitScenario, "ovphysx", gpu_device())


class TestRigidBodyViewStandaloneRecursiveUnderCluster:
    """Resolve every non-articulated cluster body through recursive descent."""

    def test_rigid_body_view_standalone_recursive_under_cluster_ovphysx_cc(self) -> None:
        """Resolve recursive cluster descendants with CPU simulation and CPU tensors."""
        run_scenario(self, _StandaloneRecursiveScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_rigid_body_view_standalone_recursive_under_cluster_ovphysx_gg(self) -> None:
        """Resolve recursive cluster descendants with GPU simulation and GPU tensors."""
        run_scenario(self, _StandaloneRecursiveScenario, "ovphysx", gpu_device())
