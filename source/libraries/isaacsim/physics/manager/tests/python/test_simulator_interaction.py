# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Verify manager aggregation of interactive raycasts and prim debug data."""

from __future__ import annotations

import _physics_setup  # noqa: F401 - initializes Carbonite framework
import isaacsim.physics.manager as physics_manager
import isaacsim.physics.registration as physics_registration
from isaacsim.physics.manager.impl import (
    DebugDataItemType,
    Float3,
    Simulation,
    k_invalid_simulation_id,
)

debug_data_entries: dict[str, dict[str, object]] = {
    "test float": {"type": DebugDataItemType.FLOAT, "value": 0.0},
    "test int": {"type": DebugDataItemType.INT, "value": 0},
    "test string": {"type": DebugDataItemType.STRING, "value": "test"},
    "test bool": {"type": DebugDataItemType.BOOL, "value": True},
    "test vector3": {"type": DebugDataItemType.VECTOR, "value": (1.0, 2.0, 3.0)},
    "test point3": {"type": DebugDataItemType.POINT, "value": (1.0, 2.0, 3.0)},
    "test quaternion": {"type": DebugDataItemType.QUATERNION, "value": (1.0, 2.0, 3.0, 4.0)},
}


class MockInteraction:
    """Test double that records raycasts and returns deterministic debug data."""

    def __init__(self) -> None:
        self.raycast_count = 0
        self.last_raycast_input = False

    def handle_raycast(self, origin: Float3, direction: Float3, has_input: bool) -> bool:
        """Record one interactive raycast request.

        Args:
            origin: Ray origin.
            direction: Ray direction.
            has_input: Whether interaction input is active.

        Returns:
            True because the mock accepts every request.

        """
        del origin, direction
        self.raycast_count += 1
        self.last_raycast_input = has_input
        return True

    def get_prim_debug_data(self, prim_path: str) -> dict[str, dict[str, object]]:
        """Get deterministic debug entries for a prim.

        Args:
            prim_path: Prim path requested by the manager.

        Returns:
            Shared mapping of typed mock debug entries.

        """
        del prim_path
        return debug_data_entries


class TestSimulateInteraction:
    """Verify interaction callbacks through the physics manager."""

    def setup_method(self) -> None:
        """Register a fresh interaction mock before each test."""
        self.physics = physics_registration
        self.physics_interaction = physics_manager
        self.mock = MockInteraction()

        self.simulation = Simulation()
        self.simulation.interaction_fns.handle_raycast = self.mock.handle_raycast
        self.simulation.interaction_fns.get_prim_debug_data = self.mock.get_prim_debug_data

        self.simulation_id = self.physics.register_simulation(self.simulation, "MockInteractionSim")
        assert self.simulation_id != k_invalid_simulation_id

    def teardown_method(self) -> None:
        """Unregister the interaction mock after each test."""
        if self.simulation_id != k_invalid_simulation_id:
            self.physics.unregister_simulation(self.simulation_id)

    def test_raycast_handling(self) -> None:
        """Verify that raycasts preserve input state and reach the backend."""
        origin = Float3(1.0, 2.0, 3.0)
        direction = Float3(0.0, 1.0, 0.0)

        self.physics_interaction.handle_raycast(origin, direction, True)
        assert self.mock.raycast_count == 1
        assert self.mock.last_raycast_input

        self.physics_interaction.handle_raycast(origin, direction, False)
        assert self.mock.raycast_count == 2
        assert not (self.mock.last_raycast_input)

    def test_get_prim_debug_data(self) -> None:
        """Verify that manager debug data matches every backend-provided entry."""
        debug_data = self.physics_interaction.get_prim_debug_data("/World/Cube")
        assert set(debug_data.keys()) == set(debug_data_entries.keys())
        for key, entry in debug_data.items():
            golden = debug_data_entries[key]
            for item_key, item_value in entry.items():
                assert item_value == golden[item_key], f"Mismatch for key {key}, item {item_key}"
