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

"""Validate heterogeneous articulation batches with the OvPhysX engine."""

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
from common.heterogeneous import (  # noqa: E402
    HeterogeneousBaseDynamicsCommon,
    HeterogeneousSceneArticulationsCommon,
)


class TestHeterogeneousBaseArticulationsDynamics:
    """Compare CartPole and Ant state and dynamics tensors across replicas."""

    def test_heterogeneous_base_articulation_dynamics_ovphysx_cc(self) -> None:
        """Compare mixed articulation dynamics with CPU simulation and CPU tensors."""
        run_scenario(self, HeterogeneousBaseDynamicsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_heterogeneous_base_articulation_dynamics_ovphysx_gg(self) -> None:
        """Compare mixed articulation dynamics with GPU simulation and GPU tensors."""
        run_scenario(self, HeterogeneousBaseDynamicsCommon, "ovphysx", gpu_device())


class TestHeterogeneousSceneArticulations:
    """Validate cabinet and Franka link transforms in one scene.

    The scenario compares environment-local link transforms across all
    replicated environments.
    """

    def test_heterogeneous_scene_articulation_ovphysx_cc(self) -> None:
        """Compare cabinet and Franka link poses with CPU simulation and CPU tensors."""
        run_scenario(self, HeterogeneousSceneArticulationsCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_heterogeneous_scene_articulation_ovphysx_gg(self) -> None:
        """Compare cabinet and Franka link poses with GPU simulation and GPU tensors."""
        run_scenario(self, HeterogeneousSceneArticulationsCommon, "ovphysx", gpu_device())
