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

"""Validate OvPhysX signed-distance-field shape evaluation.

The GPU scenario compares distances and gradients against an analytic box SDF
and validates caller-provided output buffers. The CPU scenario requires the
view to report itself as unsupported.
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
from common.sdf_shape import (  # noqa: E402
    SdfDistancesAndGradientsCommon,
    SdfQueryCapacityResizeCommon,
)


class TestSdfShapeView:
    """Validate analytic box distances, gradients, and output-buffer guards."""

    def test_sdf_shapes_ovphysx_cc(self) -> None:
        """Require unsupported SDF reporting with CPU simulation and CPU tensors."""
        run_scenario(self, SdfDistancesAndGradientsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_sdf_shapes_ovphysx_gg(self) -> None:
        """Check analytic box SDF queries with GPU simulation and GPU tensors."""
        run_scenario(self, SdfDistancesAndGradientsCommon, "ovphysx", gpu_device())


class TestSdfQueryCapacityResize:
    """Validate that an SDF view adopts the query extent the caller submits."""

    def test_sdf_query_capacity_resize_ovphysx_cc(self) -> None:
        """Check the CPU lane, where the view is unsupported and nothing resizes."""
        run_scenario(self, SdfQueryCapacityResizeCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_sdf_query_capacity_resize_ovphysx_gg(self) -> None:
        """Check grow and shrink of the query extent on the GPU lane."""
        run_scenario(self, SdfQueryCapacityResizeCommon, "ovphysx", gpu_device())
