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

"""Validate simulation-view tensors with the OvPhysX engine."""

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
from common.simulation_view import (  # noqa: E402
    CreateEntityConstructionParityCommon,
    GravityCommon,
)


class TestSimulationViewGravity:
    """Round-trip a unit positive-Z simulation gravity vector."""

    def test_simulation_view_gravity_ovphysx_cc(self) -> None:
        """Round-trip gravity with CPU simulation and CPU tensors."""
        run_scenario(self, GravityCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_simulation_view_gravity_ovphysx_gg(self) -> None:
        """Round-trip gravity with GPU simulation and GPU tensors."""
        run_scenario(self, GravityCommon, "ovphysx", gpu_device())


class TestDataPaths:
    """Verify that required simulation-view test assets exist."""

    @pytest.mark.parametrize(
        "asset",
        ("Ant.usda", "CartPole.usda", "CartPoleNoRail.usda", "CartRailNoPole.usda", "Humanoid.usda"),
    )
    def test_data_paths(self, asset: str) -> None:
        """Verify that a required scenario asset exists.

        Args:
            asset: Asset file name supplied by the pytest parameter.
        """
        from _scenario import get_asset_root

        asset_root = get_asset_root()
        assert os.path.isdir(asset_root)
        assert os.path.isfile(os.path.join(asset_root, asset)), f"missing test asset {asset!r}"


class TestCreateEntityConstructionParity:
    """Verify create_entity and SimulationView.create_*_view agree against a live simulation."""

    def test_create_entity_construction_parity_ovphysx_cc(self) -> None:
        """Verify construction parity on the ovphysx CPU pipeline."""
        run_scenario(self, CreateEntityConstructionParityCommon, "ovphysx", cpu_device())

    @gpu_only
    def test_create_entity_construction_parity_ovphysx_gg(self) -> None:
        """Verify construction parity on the ovphysx GPU pipeline."""
        run_scenario(self, CreateEntityConstructionParityCommon, "ovphysx", gpu_device())
