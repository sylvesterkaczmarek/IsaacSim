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

"""Declare unimplemented Newton deformable-material tensor scenarios."""

from __future__ import annotations

import os
import sys

import _physics_setup  # noqa: F401

_TENSORS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TENSORS_DIR not in sys.path:
    sys.path.append(_TENSORS_DIR)

from _legacy_runner import unimplemented_placeholder  # noqa: E402


@unimplemented_placeholder
class TestDeformableMaterialProperties:
    """Record unimplemented Newton deformable-material property scenarios."""

    def test_youngs_modulus_newton(self) -> None:
        """Record the unimplemented Young's-modulus case for Newton."""

    def test_poissons_ratio_newton(self) -> None:
        """Record the unimplemented Poisson's-ratio case for Newton."""

    def test_density_newton(self) -> None:
        """Record the unimplemented density case for Newton."""

    def test_dynamic_friction_newton(self) -> None:
        """Record the unimplemented dynamic friction case for Newton."""

    def test_elasticity_damping_newton(self) -> None:
        """Record the unimplemented elasticity damping case for Newton."""

    def test_bend_stiffness_newton(self) -> None:
        """Record the unimplemented bend stiffness case for Newton."""

    def test_thickness_newton(self) -> None:
        """Record the unimplemented thickness case for Newton."""

    def test_bend_damping_newton(self) -> None:
        """Record the unimplemented bend damping case for Newton."""
