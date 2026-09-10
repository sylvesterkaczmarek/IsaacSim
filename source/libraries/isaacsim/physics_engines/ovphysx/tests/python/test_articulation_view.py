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

"""Validate articulation-view tensors with the OvPhysX engine.

The special-case and duplicate-name scenarios add PhysX articulation settings
to their engine-neutral USD fixtures. The remaining scenarios exercise the
fixtures without additional engine-specific authoring.
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
from common.articulation_view import (  # noqa: E402
    ArticulationCentroidalMomentumAndMassCommon,
    ArticulationDofPropertiesCommon,
    ArticulationSpecialCasesCommon,
    ArticulationViewCommon,
    ArticulationViewDuplicateNamesCommon,
    HumanoidViewCommon,
)
from physx_usd_schemas import PhysxSchema  # noqa: E402

# ---------------------------------------------------------------------------
# ovphysx engine-specific extension.
# ---------------------------------------------------------------------------


class _PhysxArticulationApiMixin:
    """Configure each simple articulation for deterministic tensor tests.

    The mixin disables sleeping and self-collision on every replicated
    articulation root.
    """

    def _apply_engine_specifics(self) -> None:
        super()._apply_engine_specifics()
        for i in range(self.num_envs):
            prim = self.stage.GetPrimAtPath(f"/envs/env{i}/SimpleArticulation")
            if prim:
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateSleepThresholdAttr(0.0)
                api.CreateEnabledSelfCollisionsAttr().Set(False)


class _ArticulationSpecialCasesScenario(_PhysxArticulationApiMixin, ArticulationSpecialCasesCommon):
    pass


# ---------------------------------------------------------------------------
# Test classes.
# ---------------------------------------------------------------------------


class TestArticulationView:
    """Validate CartPole articulation count and resolved prim-path metadata."""

    def test_articulation_view_ovphysx_cc(self) -> None:
        """Check CartPole view discovery with CPU simulation and CPU tensors."""
        run_scenario(self, ArticulationViewCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_view_ovphysx_gg(self) -> None:
        """Check CartPole view discovery with GPU simulation and GPU tensors."""
        run_scenario(self, ArticulationViewCommon, "ovphysx", gpu_device())


class TestHumanoidView:
    """Validate the count, link count, and DOF count of a Humanoid view."""

    def test_humanoid_view_ovphysx_cc(self) -> None:
        """Check Humanoid view dimensions with CPU simulation and CPU tensors."""
        run_scenario(self, HumanoidViewCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_humanoid_view_ovphysx_gg(self) -> None:
        """Check Humanoid view dimensions with GPU simulation and GPU tensors."""
        run_scenario(self, HumanoidViewCommon, "ovphysx", gpu_device())


class TestArticulationDofProperties:
    """Round-trip Humanoid joint limits and drive properties."""

    def test_articulation_dof_properties_ovphysx_cc(self) -> None:
        """Round-trip Humanoid DOF properties with CPU simulation and CPU tensors."""
        run_scenario(self, ArticulationDofPropertiesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_dof_properties_ovphysx_gg(self) -> None:
        """Round-trip Humanoid DOF properties with GPU simulation and GPU tensors."""
        run_scenario(self, ArticulationDofPropertiesCommon, "ovphysx", gpu_device())


class TestArticulationSpecialCases:
    """Validate tensor access for a single-link articulation with zero DOFs.

    The scenario exercises the ``(count, 0)`` shapes produced by all zero-DOF
    accessors.
    """

    def test_articulation_special_cases_ovphysx_cc(self) -> None:
        """Round-trip empty DOF tensors with CPU simulation and CPU tensors."""
        run_scenario(self, _ArticulationSpecialCasesScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_special_cases_ovphysx_gg(self) -> None:
        """Round-trip empty DOF tensors with GPU simulation and GPU tensors."""
        run_scenario(self, _ArticulationSpecialCasesScenario, "ovphysx", gpu_device())


class _ArticulationViewDuplicateNamesScenario(_PhysxArticulationApiMixin, ArticulationViewDuplicateNamesCommon):
    pass


class TestArticulationViewDuplicateNames:
    """Require unique link and joint metadata for duplicate topology leaf names."""

    def test_articulation_view_duplicate_name_ovphysx_cc(self) -> None:
        """Check unique topology names with CPU simulation and CPU tensors."""
        run_scenario(self, _ArticulationViewDuplicateNamesScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_view_duplicate_name_ovphysx_gg(self) -> None:
        """Check unique topology names with GPU simulation and GPU tensors."""
        run_scenario(self, _ArticulationViewDuplicateNamesScenario, "ovphysx", gpu_device())


class TestArticulationCentroidalMomentumAndMass:
    """Validate articulation mass-center and centroidal-momentum reads.

    The scenarios require both tensor operations to produce correctly shaped
    data for every replicated articulation.
    """

    def test_articulation_centroidal_momentum_and_mass_ovphysx_cc(self) -> None:
        """Check Ant mass center and Humanoid momentum with CPU simulation and CPU tensors."""
        run_scenario(self, ArticulationCentroidalMomentumAndMassCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_centroidal_momentum_and_mass_ovphysx_gg(self) -> None:
        """Check Ant mass center and Humanoid momentum with GPU simulation and GPU tensors."""
        run_scenario(self, ArticulationCentroidalMomentumAndMassCommon, "ovphysx", gpu_device())
