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

"""Verify simulation registration, activation, enumeration, and events."""

from __future__ import annotations

import _physics_setup  # noqa: F401 - initializes Carbonite framework
import isaacsim.physics.registration as physics_registration
from expected_error import ExpectedError
from isaacsim.physics.manager.impl import (
    Simulation,
    SimulationRegistryEventType,
    k_invalid_simulation_id,
)


class TestSimulationRegistry:
    """Exercise the process-wide simulation registry."""

    def setup_method(self) -> None:
        """Resolve the registration API before each test."""
        self.physics = physics_registration
        assert self.physics is not None

    def test_create_new_simulation(self) -> None:
        """Verify that a registered simulation can be resolved by name."""
        simulation = Simulation()
        simulation_name = "TestSimulation"

        simulation_id = self.physics.register_simulation(simulation, simulation_name)
        assert simulation_id != k_invalid_simulation_id

        retrieved_name = self.physics.get_simulation_name(simulation_id)
        assert retrieved_name == simulation_name

        self.physics.unregister_simulation(simulation_id)

    def test_invalid_simulation_operations(self) -> None:
        """Verify safe results and diagnostics for an invalid simulation identifier."""
        with ExpectedError():
            invalid_sim = self.physics.get_simulation(k_invalid_simulation_id)
            assert invalid_sim is None

        with ExpectedError():
            invalid_name = self.physics.get_simulation_name(k_invalid_simulation_id)
            assert invalid_name == ""

        with ExpectedError():
            assert not (self.physics.is_simulation_active(k_invalid_simulation_id))

        with ExpectedError():
            self.physics.unregister_simulation(k_invalid_simulation_id)

        with ExpectedError():
            self.physics.activate_simulation(k_invalid_simulation_id)

        with ExpectedError():
            self.physics.deactivate_simulation(k_invalid_simulation_id)

    def test_register_and_unregister_simulation(self) -> None:
        """Verify simulation lookup before and after unregistration."""
        simulation = Simulation()
        simulation_name = "TestSimulation"

        simulation_id = self.physics.register_simulation(simulation, simulation_name)
        assert simulation_id != k_invalid_simulation_id

        retrieved_sim = self.physics.get_simulation(simulation_id)
        assert retrieved_sim is not None

        retrieved_name = self.physics.get_simulation_name(simulation_id)
        assert retrieved_name == simulation_name

        self.physics.unregister_simulation(simulation_id)
        with ExpectedError():
            retrieved_sim = self.physics.get_simulation(simulation_id)
            assert retrieved_sim is None

    def test_multiple_simulations_management(self) -> None:
        """Verify enumeration and names for three concurrently registered simulations."""
        sim1 = Simulation()
        sim2 = Simulation()
        sim3 = Simulation()

        id1 = self.physics.register_simulation(sim1, "Simulation1")
        id2 = self.physics.register_simulation(sim2, "Simulation2")
        id3 = self.physics.register_simulation(sim3, "Simulation3")

        assert id1 != k_invalid_simulation_id
        assert id2 != k_invalid_simulation_id
        assert id3 != k_invalid_simulation_id

        # Count includes any auto-registered backends (e.g. ovphysx); check
        # relative to the number registered by this test, not absolute.
        num_sims = self.physics.get_num_simulations()
        assert num_sims >= 3

        sim_ids = self.physics.get_simulation_ids()
        assert len(sim_ids) == num_sims

        found_id1 = found_id2 = found_id3 = False
        for sim_id in sim_ids:
            if sim_id == id1:
                found_id1 = True
                assert self.physics.get_simulation_name(sim_id) == "Simulation1"
            if sim_id == id2:
                found_id2 = True
                assert self.physics.get_simulation_name(sim_id) == "Simulation2"
            if sim_id == id3:
                found_id3 = True
                assert self.physics.get_simulation_name(sim_id) == "Simulation3"

        assert found_id1
        assert found_id2
        assert found_id3

        self.physics.unregister_simulation(id1)
        self.physics.unregister_simulation(id2)
        self.physics.unregister_simulation(id3)

    def test_simulation_activation(self) -> None:
        """Verify simulation deactivation and reactivation."""
        simulation = Simulation()
        simulation_id = self.physics.register_simulation(simulation, "TestSimulation")
        assert simulation_id != k_invalid_simulation_id

        assert self.physics.is_simulation_active(simulation_id)

        self.physics.deactivate_simulation(simulation_id)
        assert not (self.physics.is_simulation_active(simulation_id))

        self.physics.activate_simulation(simulation_id)
        assert self.physics.is_simulation_active(simulation_id)

        self.physics.unregister_simulation(simulation_id)

    def _on_simulation_registry_event(
        self,
        event_type: SimulationRegistryEventType,
        simulation_id: physics_registration.SimulationId,
        simulation_name: str,
    ) -> None:
        """Record the latest simulation registry event.

        Args:
            event_type: Registry operation that emitted the event.
            simulation_id: Identifier of the affected simulation.
            simulation_name: Name of the affected simulation.

        """
        self.event_type = event_type
        self.simulation_id = simulation_id
        self.simulation_name = simulation_name

    def test_simulation_registry_events(self) -> None:
        """Verify registration, activation, deactivation, and unregistration events."""
        simulation = Simulation()
        simulation_name = "TestSimulation"

        subscription = self.physics.subscribe_simulation_registry_events(self._on_simulation_registry_event)

        simulation_id = self.physics.register_simulation(simulation, simulation_name)
        assert simulation_id != k_invalid_simulation_id
        assert self.simulation_id == simulation_id
        assert self.simulation_name == simulation_name
        assert self.event_type == SimulationRegistryEventType.SIMULATION_REGISTERED

        self.physics.deactivate_simulation(simulation_id)
        assert self.event_type == SimulationRegistryEventType.SIMULATION_DEACTIVATED

        self.physics.activate_simulation(simulation_id)
        assert self.event_type == SimulationRegistryEventType.SIMULATION_ACTIVATED

        self.physics.unregister_simulation(simulation_id)
        assert self.event_type == SimulationRegistryEventType.SIMULATION_UNREGISTERED

        # Unsubscribe and verify no more events
        subscription = None
        self.simulation_id = k_invalid_simulation_id
        self.simulation_name = ""
        self.event_type = None

        simulation_id = self.physics.register_simulation(simulation, simulation_name)
        assert simulation_id != k_invalid_simulation_id
        assert self.simulation_id == k_invalid_simulation_id
        assert self.simulation_name == ""
        assert self.event_type is None

        self.physics.unregister_simulation(simulation_id)
