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

"""Validate articulation property tensors with the OvPhysX engine."""

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
from common.articulation_properties import (  # noqa: E402
    BodyPropertiesCommon,
    FixedTendonPropertiesCommon,
    HeterogeneousFixedTendonPropertiesCommon,
    HeterogeneousSpatialTendonPropertiesCommon,
    ShapePropertiesCommon,
    SpatialTendonPropertiesCommon,
)


class TestArticulationBodyProperties:
    """Round-trip Humanoid link masses, centers of mass, and inertias."""

    def test_articulation_body_properties_ovphysx_cc(self) -> None:
        """Round-trip Humanoid body properties with CPU simulation and CPU tensors."""
        run_scenario(self, BodyPropertiesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_body_properties_ovphysx_gg(self) -> None:
        """Round-trip Humanoid body properties with GPU simulation and GPU tensors."""
        run_scenario(self, BodyPropertiesCommon, "ovphysx", gpu_device())


class TestArticulationShapeProperties:
    """Round-trip Humanoid material and collision-offset properties."""

    def test_articulation_shape_properties_ovphysx_cc(self) -> None:
        """Round-trip Humanoid shape properties with CPU simulation and CPU tensors."""
        run_scenario(self, ShapePropertiesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_shape_properties_ovphysx_gg(self) -> None:
        """Round-trip Humanoid shape properties with GPU simulation and GPU tensors."""
        run_scenario(self, ShapePropertiesCommon, "ovphysx", gpu_device())


class TestArticulationFixedTendonProperties:
    """Round-trip all fixed-tendon properties on replicated Shadow Hands."""

    def test_articulation_fixed_tendon_properties_ovphysx_cc(self) -> None:
        """Round-trip fixed-tendon properties with CPU simulation and CPU tensors."""
        run_scenario(self, FixedTendonPropertiesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_fixed_tendon_properties_ovphysx_gg(self) -> None:
        """Round-trip fixed-tendon properties with GPU simulation and GPU tensors."""
        run_scenario(self, FixedTendonPropertiesCommon, "ovphysx", gpu_device())


class TestArticulationSpatialTendonProperties:
    """Round-trip all properties on replicated spatial tendons."""

    def test_articulation_spatial_tendon_properties_ovphysx_cc(self) -> None:
        """Round-trip spatial-tendon properties with CPU simulation and CPU tensors."""
        run_scenario(self, SpatialTendonPropertiesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_spatial_tendon_properties_ovphysx_gg(self) -> None:
        """Round-trip spatial-tendon properties with GPU simulation and GPU tensors."""
        run_scenario(self, SpatialTendonPropertiesCommon, "ovphysx", gpu_device())


class TestArticulationHeterogeneousFixedTendonProperties:
    """Validate fixed-tendon properties across two articulation topologies."""

    def test_articulation_heterogeneous_fixed_tendon_properties_ovphysx_cc(self) -> None:
        """Check heterogeneous fixed tendons with CPU simulation and CPU tensors."""
        run_scenario(self, HeterogeneousFixedTendonPropertiesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_heterogeneous_fixed_tendon_properties_ovphysx_gg(self) -> None:
        """Check heterogeneous fixed tendons with GPU simulation and GPU tensors."""
        run_scenario(self, HeterogeneousFixedTendonPropertiesCommon, "ovphysx", gpu_device())


class TestArticulationHeterogeneousSpatialTendonProperties:
    """Validate spatial-tendon properties across two articulation topologies."""

    def test_articulation_heterogeneous_spatial_tendon_properties_ovphysx_cc(self) -> None:
        """Check heterogeneous spatial tendons with CPU simulation and CPU tensors."""
        run_scenario(self, HeterogeneousSpatialTendonPropertiesCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_articulation_heterogeneous_spatial_tendon_properties_ovphysx_gg(self) -> None:
        """Check heterogeneous spatial tendons with GPU simulation and GPU tensors."""
        run_scenario(self, HeterogeneousSpatialTendonPropertiesCommon, "ovphysx", gpu_device())
