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

"""Validate articulation-view tensors with the Newton engine.

The engine-neutral scenarios run directly because Newton does not require the
PhysX-specific articulation schema settings used by the OvPhysX fixtures.
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
from common.articulation_view import (  # noqa: E402
    ArticulationCentroidalMomentumAndMassCommon,
    ArticulationDofLimitsSubsetCommon,
    ArticulationDofPropertiesCommon,
    ArticulationSpecialCasesCommon,
    ArticulationViewCommon,
    ArticulationViewDuplicateNamesCommon,
    HumanoidViewCommon,
)

# Contract cases blocked on Newton ports not yet done (impl registration / metatype
# name uniquification). Skipped explicitly with a reason -- the runner re-raises, so
# an unported impl would otherwise surface as a hard failure rather than a skip.
_GAP_VARWIDTH_DOF = (
    "Newton DOF-stiffness operations require uniform, dense DOF rows; "
    "variable-width articulations have no rectangular DOF map"
)
_GAP_MASS_CENTER = "Newton exposes no aggregate mass-center or centroidal-momentum tensor operation"


class TestArticulationView:
    """Validate articulation view with Newton."""

    def test_articulation_view_newton_cc(self) -> None:
        """Verify articulation view on the Newton CPU pipeline."""
        run_scenario(self, ArticulationViewCommon, "newton", cpu_device())

    @gpu_only
    def test_articulation_view_newton_gg(self) -> None:
        """Verify articulation view on the Newton GPU pipeline."""
        run_scenario(self, ArticulationViewCommon, "newton", gpu_device())


class TestHumanoidView:
    """Validate humanoid view with Newton."""

    def test_humanoid_view_newton_cc(self) -> None:
        """Verify humanoid view on the Newton CPU pipeline."""
        run_scenario(self, HumanoidViewCommon, "newton", cpu_device())

    @gpu_only
    def test_humanoid_view_newton_gg(self) -> None:
        """Verify humanoid view on the Newton GPU pipeline."""
        run_scenario(self, HumanoidViewCommon, "newton", gpu_device())


class TestArticulationDofProperties:
    """Validate articulation DOF properties with Newton."""

    def test_articulation_dof_properties_newton_cc(self) -> None:
        """Verify articulation DOF properties on the Newton CPU pipeline."""
        run_scenario(self, ArticulationDofPropertiesCommon, "newton", cpu_device())

    @gpu_only
    def test_articulation_dof_properties_newton_gg(self) -> None:
        """Verify articulation DOF properties on the Newton GPU pipeline."""
        run_scenario(self, ArticulationDofPropertiesCommon, "newton", gpu_device())


class TestArticulationDofLimitsSubset:
    """Validate articulation DOF limits subset with Newton."""

    def test_articulation_dof_limits_subset_newton_cc(self) -> None:
        """Verify articulation DOF limits subset on the Newton CPU pipeline."""
        run_scenario(self, ArticulationDofLimitsSubsetCommon, "newton", cpu_device())

    @gpu_only
    def test_articulation_dof_limits_subset_newton_gg(self) -> None:
        """Verify articulation DOF limits subset on the Newton GPU pipeline."""
        run_scenario(self, ArticulationDofLimitsSubsetCommon, "newton", gpu_device())


class TestArticulationSpecialCases:
    """Skip zero-DOF articulation checks because Newton does not support variable widths."""

    @pytest.mark.skip(reason=_GAP_VARWIDTH_DOF)
    def test_articulation_special_cases_newton_cc(self) -> None:
        """Verify articulation special cases on the Newton CPU pipeline."""
        run_scenario(self, ArticulationSpecialCasesCommon, "newton", cpu_device())

    @pytest.mark.skip(reason=_GAP_VARWIDTH_DOF)
    @gpu_only
    def test_articulation_special_cases_newton_gg(self) -> None:
        """Verify articulation special cases on the Newton GPU pipeline."""
        run_scenario(self, ArticulationSpecialCasesCommon, "newton", gpu_device())


class TestArticulationViewDuplicateNames:
    """Validate articulation view duplicate names with Newton."""

    def test_articulation_view_duplicate_name_newton_cc(self) -> None:
        """Verify articulation view duplicate name on the Newton CPU pipeline."""
        run_scenario(self, ArticulationViewDuplicateNamesCommon, "newton", cpu_device())

    @gpu_only
    def test_articulation_view_duplicate_name_newton_gg(self) -> None:
        """Verify articulation view duplicate name on the Newton GPU pipeline."""
        run_scenario(self, ArticulationViewDuplicateNamesCommon, "newton", gpu_device())


class TestArticulationCentroidalMomentumAndMass:
    """Skip centroidal data checks because Newton lacks mass-center operations."""

    @pytest.mark.skip(reason=_GAP_MASS_CENTER)
    def test_articulation_centroidal_momentum_and_mass_newton_cc(self) -> None:
        """Verify articulation centroidal momentum and mass on the Newton CPU pipeline."""
        run_scenario(self, ArticulationCentroidalMomentumAndMassCommon, "newton", cpu_device())

    @pytest.mark.skip(reason=_GAP_MASS_CENTER)
    @gpu_only
    def test_articulation_centroidal_momentum_and_mass_newton_gg(self) -> None:
        """Verify articulation centroidal momentum and mass on the Newton GPU pipeline."""
        run_scenario(self, ArticulationCentroidalMomentumAndMassCommon, "newton", gpu_device())
