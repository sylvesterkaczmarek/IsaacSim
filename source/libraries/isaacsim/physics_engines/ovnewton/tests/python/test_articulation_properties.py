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

"""Validate articulation property tensors with the Newton engine."""

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
    unimplemented_placeholder,
)
from common.articulation_properties import (  # noqa: E402
    BodyPropertiesCommon,
    ShapePropertiesCommon,
)


class TestArticulationBodyProperties:
    """Validate articulation body properties with Newton."""

    def test_articulation_body_properties_newton_cc(self) -> None:
        """Verify articulation body properties on the Newton CPU pipeline."""
        run_scenario(self, BodyPropertiesCommon, "newton", cpu_device())

    @gpu_only
    def test_articulation_body_properties_newton_gg(self) -> None:
        """Verify articulation body properties on the Newton GPU pipeline."""
        run_scenario(self, BodyPropertiesCommon, "newton", gpu_device())


# Newton's per-shape properties surface (num-shapes / material-properties /
# contact-offsets / rest-offsets) is unwired, and Newton only partially has the data
# (single friction mu, no true contact/rest offset). Temporary gap; real wiring +
# host-cached round-trip belong in a separate Newton-completeness MR.
_GAP_NEWTON_SHAPE_PROPERTIES = "_GAP_NEWTON_SHAPE_PROPERTIES: per-shape material/offset surface unwired on Newton"


@pytest.mark.skip(reason=_GAP_NEWTON_SHAPE_PROPERTIES)
class TestArticulationShapeProperties:
    """Skip shape-property round trips because Newton lacks that tensor surface."""

    def test_articulation_shape_properties_newton_cc(self) -> None:
        """Verify articulation shape properties on the Newton CPU pipeline."""
        run_scenario(self, ShapePropertiesCommon, "newton", cpu_device())

    @gpu_only
    def test_articulation_shape_properties_newton_gg(self) -> None:
        """Verify articulation shape properties on the Newton GPU pipeline."""
        run_scenario(self, ShapePropertiesCommon, "newton", gpu_device())


# ---------------------------------------------------------------------------
# Placeholder shells — tendon-property tests don't have Newton scenarios
# yet (PhysxSchema-blocked on the ovphysx side too). Mirrored here so the
# matrix is symmetric.
# ---------------------------------------------------------------------------


@unimplemented_placeholder
class TestArticulationFixedTendonProperties:
    """Record unimplemented Newton fixed-tendon property scenarios."""

    def test_articulation_fixed_tendon_properties_newton_cc(self) -> None:
        """Record the unimplemented articulation fixed tendon properties case for Newton."""

    def test_articulation_fixed_tendon_properties_newton_gg(self) -> None:
        """Record the unimplemented articulation fixed tendon properties case for Newton."""


@unimplemented_placeholder
class TestArticulationSpatialTendonProperties:
    """Record unimplemented Newton spatial-tendon property scenarios."""

    def test_articulation_spatial_tendon_properties_newton_cc(self) -> None:
        """Record the unimplemented articulation spatial tendon properties case for Newton."""

    def test_articulation_spatial_tendon_properties_newton_gg(self) -> None:
        """Record the unimplemented articulation spatial tendon properties case for Newton."""


@unimplemented_placeholder
class TestArticulationHeterogeneousFixedTendonProperties:
    """Record unimplemented heterogeneous fixed-tendon scenarios for Newton."""

    def test_articulation_heterogeneous_fixed_tendon_properties_newton_cc(self) -> None:
        """Record the unimplemented articulation heterogeneous fixed tendon properties case for Newton."""

    def test_articulation_heterogeneous_fixed_tendon_properties_newton_gg(self) -> None:
        """Record the unimplemented articulation heterogeneous fixed tendon properties case for Newton."""


@unimplemented_placeholder
class TestArticulationHeterogeneousSpatialTendonProperties:
    """Record unimplemented heterogeneous spatial-tendon scenarios for Newton."""

    def test_articulation_heterogeneous_spatial_tendon_properties_newton_cc(self) -> None:
        """Record the unimplemented articulation heterogeneous spatial tendon properties case for Newton."""

    def test_articulation_heterogeneous_spatial_tendon_properties_newton_gg(self) -> None:
        """Record the unimplemented articulation heterogeneous spatial tendon properties case for Newton."""
