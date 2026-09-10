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

"""Validate OvPhysX deformable-material entity views."""

from __future__ import annotations

import os
import sys

import _physics_setup  # noqa: F401

_TENSORS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TENSORS_DIR not in sys.path:
    sys.path.append(_TENSORS_DIR)

from _legacy_runner import gpu_device, gpu_only, run_scenario  # noqa: E402
from common.deformable_material import DeformableMaterialPropertiesCommon  # noqa: E402


class TestDeformableMaterialProperties:
    """Validate OvPhysX deformable-material property tensors."""

    @gpu_only
    def test_deformable_material_properties_ovphysx_gg(self) -> None:
        """Round-trip all supported deformable-material properties on GPU."""
        run_scenario(self, DeformableMaterialPropertiesCommon, "ovphysx", gpu_device())
