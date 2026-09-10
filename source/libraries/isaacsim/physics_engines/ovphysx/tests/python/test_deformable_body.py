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

"""Validate OvPhysX volume and surface deformable-body entity views."""

from __future__ import annotations

import os
import sys

import _physics_setup  # noqa: F401

_TENSORS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TENSORS_DIR not in sys.path:
    sys.path.append(_TENSORS_DIR)

from _legacy_runner import gpu_device, gpu_only, run_scenario  # noqa: E402
from common.deformable_body import (  # noqa: E402
    SurfaceDeformableBodyCommon,
    VolumeDeformableBodyCommon,
)


class TestVolumeDeformableBody:
    """Validate OvPhysX volume-deformable-body tensors."""

    @gpu_only
    def test_volume_deformable_body_ovphysx_gg(self) -> None:
        """Validate volume-deformable-body tensors on the GPU pipeline."""
        run_scenario(self, VolumeDeformableBodyCommon, "ovphysx", gpu_device())


class TestSurfaceDeformableBody:
    """Validate OvPhysX surface-deformable-body tensors."""

    @gpu_only
    def test_surface_deformable_body_ovphysx_gg(self) -> None:
        """Validate surface-deformable-body tensors on the GPU pipeline."""
        run_scenario(self, SurfaceDeformableBodyCommon, "ovphysx", gpu_device())
