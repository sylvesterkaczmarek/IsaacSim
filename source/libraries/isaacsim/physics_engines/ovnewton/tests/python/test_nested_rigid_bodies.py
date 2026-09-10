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

"""Validate nested rigid-body and articulation paths with Newton.

The tests cover explicit paths, recursive patterns, wildcard patterns,
alternation, and nested articulation roots.
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
from pxr import Usd, UsdGeom  # noqa: E402


class TestArticulationViewNestedLinks:
    """Validate articulation view nested links with Newton."""

    def test_articulation_view_nested_links_newton_cc(self) -> None:
        """Verify articulation view nested links on the Newton CPU pipeline."""
        run_scenario(self, ArticulationViewNestedLinksCommon, "newton", cpu_device())

    @gpu_only
    def test_articulation_view_nested_links_newton_gg(self) -> None:
        """Verify articulation view nested links on the Newton GPU pipeline."""
        run_scenario(self, ArticulationViewNestedLinksCommon, "newton", gpu_device())


class TestArticulationViewAtNestedRootLink:
    """Validate articulation view at nested root link with Newton."""

    def test_articulation_view_at_nested_root_link_newton_cc(self) -> None:
        """Verify articulation view at nested root link on the Newton CPU pipeline."""
        run_scenario(self, ArticulationViewAtNestedRootLinkCommon, "newton", cpu_device())

    @gpu_only
    def test_articulation_view_at_nested_root_link_newton_gg(self) -> None:
        """Verify articulation view at nested root link on the Newton GPU pipeline."""
        run_scenario(self, ArticulationViewAtNestedRootLinkCommon, "newton", gpu_device())


class TestRigidBodyViewExplicitNestedPaths:
    """Validate rigid body view explicit nested paths with Newton."""

    def test_rigid_body_view_explicit_nested_paths_newton_cc(self) -> None:
        """Verify rigid body view explicit nested paths on the Newton CPU pipeline."""
        run_scenario(self, RigidBodyViewExplicitNestedPathsCommon, "newton", cpu_device())

    @gpu_only
    def test_rigid_body_view_explicit_nested_paths_newton_gg(self) -> None:
        """Verify rigid body view explicit nested paths on the Newton GPU pipeline."""
        run_scenario(self, RigidBodyViewExplicitNestedPathsCommon, "newton", gpu_device())


class TestRigidBodyViewRecursiveUnderRobot:
    """Validate rigid body view recursive under robot with Newton."""

    def test_rigid_body_view_recursive_under_robot_newton_cc(self) -> None:
        """Verify rigid body view recursive under robot on the Newton CPU pipeline."""
        run_scenario(self, RigidBodyViewRecursiveUnderRobotCommon, "newton", cpu_device())

    @gpu_only
    def test_rigid_body_view_recursive_under_robot_newton_gg(self) -> None:
        """Verify rigid body view recursive under robot on the Newton GPU pipeline."""
        run_scenario(self, RigidBodyViewRecursiveUnderRobotCommon, "newton", gpu_device())


class TestRigidBodyViewRecursiveNamedLink:
    """Validate rigid body view recursive named link with Newton."""

    def test_rigid_body_view_recursive_named_link_newton_cc(self) -> None:
        """Verify rigid body view recursive named link on the Newton CPU pipeline."""
        run_scenario(self, RigidBodyViewRecursiveNamedLinkCommon, "newton", cpu_device())

    @gpu_only
    def test_rigid_body_view_recursive_named_link_newton_gg(self) -> None:
        """Verify rigid body view recursive named link on the Newton GPU pipeline."""
        run_scenario(self, RigidBodyViewRecursiveNamedLinkCommon, "newton", gpu_device())


class TestArticulationViewRecursiveRootLink:
    """Validate articulation view recursive root link with Newton."""

    def test_articulation_view_recursive_root_link_newton_cc(self) -> None:
        """Verify articulation view recursive root link on the Newton CPU pipeline."""
        run_scenario(self, ArticulationViewRecursiveRootLinkCommon, "newton", cpu_device())

    @gpu_only
    def test_articulation_view_recursive_root_link_newton_gg(self) -> None:
        """Verify articulation view recursive root link on the Newton GPU pipeline."""
        run_scenario(self, ArticulationViewRecursiveRootLinkCommon, "newton", gpu_device())


class TestRigidBodyViewWildcardUnderRobot:
    """Validate rigid body view wildcard under robot with Newton."""

    def test_rigid_body_view_wildcard_under_robot_newton_cc(self) -> None:
        """Verify rigid body view wildcard under robot on the Newton CPU pipeline."""
        run_scenario(self, RigidBodyViewWildcardUnderRobotCommon, "newton", cpu_device())

    @gpu_only
    def test_rigid_body_view_wildcard_under_robot_newton_gg(self) -> None:
        """Verify rigid body view wildcard under robot on the Newton GPU pipeline."""
        run_scenario(self, RigidBodyViewWildcardUnderRobotCommon, "newton", gpu_device())


class TestRigidBodyViewNestedAlternation:
    """Validate rigid body view nested alternation with Newton."""

    def test_rigid_body_view_nested_alternation_newton_cc(self) -> None:
        """Verify rigid body view nested alternation on the Newton CPU pipeline."""
        run_scenario(self, RigidBodyViewNestedAlternationCommon, "newton", cpu_device())

    @gpu_only
    def test_rigid_body_view_nested_alternation_newton_gg(self) -> None:
        """Verify rigid body view nested alternation on the Newton GPU pipeline."""
        run_scenario(self, RigidBodyViewNestedAlternationCommon, "newton", gpu_device())


class TestRigidBodyViewStandaloneExplicitPaths:
    """Validate rigid body view standalone explicit paths with Newton."""

    def test_rigid_body_view_standalone_explicit_paths_newton_cc(self) -> None:
        """Verify rigid body view standalone explicit paths on the Newton CPU pipeline."""
        run_scenario(self, RigidBodyViewStandaloneExplicitPathsCommon, "newton", cpu_device())

    @gpu_only
    def test_rigid_body_view_standalone_explicit_paths_newton_gg(self) -> None:
        """Verify rigid body view standalone explicit paths on the Newton GPU pipeline."""
        run_scenario(self, RigidBodyViewStandaloneExplicitPathsCommon, "newton", gpu_device())


class TestRigidBodyViewStandaloneRecursiveUnderCluster:
    """Validate rigid body view standalone recursive under cluster with Newton."""

    def test_rigid_body_view_standalone_recursive_under_cluster_newton_cc(self) -> None:
        """Verify rigid body view standalone recursive under cluster on the Newton CPU pipeline."""
        run_scenario(self, RigidBodyViewStandaloneRecursiveUnderClusterCommon, "newton", cpu_device())

    @gpu_only
    def test_rigid_body_view_standalone_recursive_under_cluster_newton_gg(self) -> None:
        """Verify rigid body view standalone recursive under cluster on the Newton GPU pipeline."""
        run_scenario(self, RigidBodyViewStandaloneRecursiveUnderClusterCommon, "newton", gpu_device())


class TestFindMatchingPathsRecursiveLeaf:
    """Validate recursive and strict matching for a named leaf token.

    Recursive matching descends through differently named parents but stops
    descending after a leaf match, preventing nested duplicates. Strict
    matching considers only the leaf at the requested hierarchy level.
    """

    def _build_stage(self) -> Usd.Stage:
        """Build direct, nested, and repeated ``cartpole`` path candidates.

        Returns:
            In-memory stage used by the path-matching assertions.

        """
        stage = Usd.Stage.CreateInMemory()
        for path in (
            "/envs",
            "/envs/env0",
            "/envs/env0/cartpole",
            "/envs/env0/cartpole/inner",
            "/envs/env0/cartpole/inner/cartpole",  # same name nested under a match -> suppressed
            "/envs/env0/rack",
            "/envs/env0/rack/cartpole",  # nested under a non-match -> still recursed
            "/envs/env1",
            "/envs/env1/cartpole",
            "/envs/env1/rack",
            "/envs/env1/rack/cartpole",
        ):
            UsdGeom.Xform.Define(stage, path)
        return stage

    def test_named_leaf_recurses_by_default(self) -> None:
        """Verify recursive leaf matching finds nested paths without duplicates."""
        from isaacsim.physics_engines.ovnewton.impl.tensors.utils import find_matching_paths

        result = find_matching_paths(self._build_stage(), "/envs/*/cartpole")
        # Recursive matching finds direct and rack-nested cart-poles in each
        # environment, then stops below an already matched cart-pole.
        assert sorted(result) == [
            "/envs/env0/cartpole",
            "/envs/env0/rack/cartpole",
            "/envs/env1/cartpole",
            "/envs/env1/rack/cartpole",
        ]
        assert "/envs/env0/cartpole/inner/cartpole" not in result

    def test_named_leaf_strict_when_disabled(self) -> None:
        """Verify strict leaf matching returns only direct hierarchy matches."""
        from isaacsim.physics_engines.ovnewton.impl.tensors.utils import find_matching_paths

        result = find_matching_paths(self._build_stage(), "/envs/*/cartpole", recursive_leaf=False)
        # Strict per-level: only each env's direct cartpole child.
        assert sorted(result) == ["/envs/env0/cartpole", "/envs/env1/cartpole"]
