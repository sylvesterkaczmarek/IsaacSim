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

"""Verify simulation-view gravity reads and writes across backends."""

from __future__ import annotations

import os
import sys
from typing import Protocol

import pytest

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

from _scenario import (  # noqa: E402
    DeviceParams,
    GridParams,
    GridTestBase,
    SimParams,
    Transform,
    get_asset_root,
)


class _GravityView(Protocol):
    def set_gravity(self, gravity: list[float]) -> None: ...

    def get_gravity(self) -> object: ...


class GravityCommon(GridTestBase):
    """Scenario that verifies simulation-view gravity access.

    Args:
        test_case: Test instance associated with the scenario.
        device_params: Simulation and tensor device selection.

    """

    def __init__(self, test_case: object, device_params: DeviceParams) -> None:
        grid_params = GridParams(num_envs=16)
        grid_params.num_rows = grid_params.num_envs // 2
        grid_params.row_spacing = 2
        grid_params.col_spacing = 6.5
        sim_params = SimParams()
        super().__init__(test_case, grid_params, sim_params, device_params)

        asset_path = os.path.join(get_asset_root(), "Ant.usda")
        actor_path = self.env_template_path.AppendChild("ant")
        self.create_actor_from_asset(actor_path, Transform((0.0, 0.0, 10.0)), asset_path)

    def on_start(self, sim: _GravityView) -> None:
        """Set and read back a unit positive-Z gravity vector.

        Args:
            sim: Simulation view under test.

        """
        sim.set_gravity([0.0, 0.0, 1.0])
        gravity = sim.get_gravity()

        if hasattr(gravity, "x"):
            g = (gravity.x, gravity.y, gravity.z)
        else:
            g = tuple(gravity)
        assert g[0] == pytest.approx(0.0, rel=0.0, abs=1e-7)
        assert g[1] == pytest.approx(0.0, rel=0.0, abs=1e-7)
        assert g[2] == pytest.approx(1.0, rel=0.0, abs=1e-7)

        self.finish()

    def on_physics_step(self, sim: object, stepno: int, dt: float) -> None:
        """Accept the unused per-step callback required by the scenario harness.

        Args:
            sim: Simulation view under test.
            stepno: Zero-based simulation step number.
            dt: Simulated time interval in seconds.

        """


class CreateEntityConstructionParityCommon(GridTestBase):
    """Scenario that verifies both entity-construction paths agree against a live simulation.

    ``TensorRegistry.create_entity`` resolves the active simulation and calls the same internal factory
    ``SimulationView.create_*_view`` uses, so the two cannot disagree unless a second implementation is
    reintroduced. Checking that needs a simulation with an initialized model, which is why this is a
    scenario rather than a registration-only test.

    Args:
        test_case: Test instance associated with the scenario.
        device_params: Simulation and tensor device selection.

    """

    def __init__(self, test_case: object, device_params: DeviceParams) -> None:
        grid_params = GridParams(num_envs=4)
        grid_params.num_rows = 2
        sim_params = SimParams()
        super().__init__(test_case, grid_params, sim_params, device_params)

        asset_path = os.path.join(get_asset_root(), "Ant.usda")
        actor_path = self.env_template_path.AppendChild("ant")
        self.create_actor_from_asset(actor_path, Transform((0.0, 0.0, 10.0)), asset_path)

    def on_start(self, sim: object) -> None:
        """Compare the impl sets and support flags produced by the two construction paths.

        Args:
            sim: Simulation view under test.

        """
        import isaacsim.physics.manager.impl.tensors as t

        # Read the engine off the live view rather than the scenario: it names the engine that actually
        # built it, which is the one create_entity has to agree with.
        engine = sim.engine

        # A path list, not a bare string: the list form resolves patterns eagerly while the string form
        # stays in pattern mode, and create_entity only has the list form to offer.
        paths = ["/envs/*/ant"]
        builders = {
            "articulation": sim.create_articulation_view,
            "rigid-body": sim.create_rigid_body_view,
        }
        for entity, create_view in builders.items():
            entity_view = t.create_entity(engine, entity, paths)
            simulation_built = create_view(paths)
            entity_impls = set(entity_view.list_impls(t.ImplKind.Get))
            simulation_impls = set(simulation_built.list_impls(t.ImplKind.Get))
            assert entity_impls == simulation_impls, (
                f"{engine}/{entity}: create_entity and create_*_view registered different get impls "
                f"({sorted(entity_impls ^ simulation_impls)} differ)"
            )
            assert entity_impls, f"{engine}/{entity}: neither path registered a get implementation"
            for impl in entity_impls:
                assert entity_view.has_impl(impl, t.ImplKind.Get) == simulation_built.has_impl(
                    impl, t.ImplKind.Get
                ), f"{engine}/{entity}: the two paths disagree on support for {impl!r}"

        self.finish()

    def on_physics_step(self, sim: object, stepno: int, dt: float) -> None:
        """Accept the unused per-step callback required by the scenario harness.

        Args:
            sim: Simulation view under test.
            stepno: Zero-based simulation step number.
            dt: Simulated time interval in seconds.

        """
