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

"""Verify physics-manager simulation callbacks, data types, and capabilities."""

from __future__ import annotations

from collections.abc import Callable

import _physics_setup  # noqa: F401 - initializes Carbonite framework
import isaacsim.physics.manager as physics_manager
import isaacsim.physics.registration as physics_registration
import pytest
from isaacsim.physics.manager.impl import (
    ContactData,
    ContactDataVector,
    ContactEventHeader,
    ContactEventHeaderVector,
    ContactEventType,
    Float3,
    ForceMode,
    FrictionAnchorsDataVector,
    PhysicsStepContext,
    Simulation,
    k_invalid_simulation_id,
)

StepCallback = Callable[[float, PhysicsStepContext], None]
ContactCallback = Callable[[ContactEventHeaderVector, ContactDataVector, FrictionAnchorsDataVector], None]


class MockSimulator:
    """Observable test double for the simulation callback group."""

    def __init__(self) -> None:
        self.attached_stage_id = 0
        self.change_tracking_paused = False
        self.time_steps_per_second = 60
        self.simulation_timestamp = 0
        self.simulation_step_count = 0
        self.step_event_callbacks: list[StepCallback] = []
        self.contact_event_callbacks: list[ContactCallback] = []
        self.simulation_id = 0
        self.capability_check_enabled = True
        self.supported_capabilities: dict[str, bool] = {}

    def initialize(self, ovstage: int, usd_identifier: str) -> bool:
        """Attach the stage identifier supplied by the manager.

        Args:
            ovstage: Native stage handle, which the mock ignores.
            usd_identifier: Integer stage identifier encoded as text.

        Returns:
            True because the mock always accepts initialization.

        """
        del ovstage  # mock ignores the ovstage; the id roundtrips via the identifier
        self.attached_stage_id = int(usd_identifier) if usd_identifier else 0
        return True

    def close(self) -> bool:
        """Clear the attached stage and report success.

        Returns:
            True because the mock always closes successfully.

        """
        self.attached_stage_id = 0
        return True

    def get_attached_stage(self) -> int:
        """Get the attached stage identifier.

        Returns:
            Current stage identifier, or zero when closed.

        """
        return self.attached_stage_id

    def simulate_async(self, elapsed_time: float, current_time: float) -> None:
        """Record one step and notify step subscribers.

        Args:
            elapsed_time: Simulated time interval in seconds.
            current_time: Simulation time before the step.

        """
        del current_time
        self.simulation_timestamp += 1
        self.simulation_step_count += 1
        context = PhysicsStepContext()
        context.scene_path = self.attached_stage_id
        context.simulation_id = self.simulation_id
        for callback in self.step_event_callbacks:
            callback(elapsed_time, context)

    def simulate(self, elapsed_time: float, current_time: float) -> None:
        """Record one synchronous step and publish its results.

        Args:
            elapsed_time: Simulated time interval in seconds.
            current_time: Simulation time before the step.

        """
        self.simulate_async(elapsed_time, current_time)
        self.fetch_results()

    def fetch_results(self) -> None:
        """Publish empty contact vectors to contact subscribers."""
        contact_event_headers = ContactEventHeaderVector()
        contact_data = ContactDataVector()
        friction_anchors = FrictionAnchorsDataVector()
        for callback in self.contact_event_callbacks:
            callback(contact_event_headers, contact_data, friction_anchors)

    def check_results(self) -> bool:
        """Report that mock results are valid.

        Returns:
            True because mock result validation cannot fail.

        """
        return True

    def flush_changes(self) -> None:
        """Accept a request to flush pending changes."""

    def pause_change_tracking(self, pause: bool) -> None:
        """Set the mock change-tracking state.

        Args:
            pause: Whether change tracking should be paused.

        """
        self.change_tracking_paused = pause

    def is_change_tracking_paused(self) -> bool:
        """Check whether change tracking is paused.

        Returns:
            Current change-tracking state.

        """
        return self.change_tracking_paused

    def subscribe_physics_contact_report_events(self, on_event: ContactCallback) -> int:
        """Register a contact-report callback.

        Args:
            on_event: Function notified with contact vectors.

        Returns:
            One-based subscription identifier.

        """
        self.contact_event_callbacks.append(on_event)
        return len(self.contact_event_callbacks)

    def unsubscribe_physics_contact_report_events(self, subscription_id: int) -> None:
        """Remove a contact-report callback.

        Args:
            subscription_id: One-based identifier returned during subscription.

        """
        if subscription_id > 0 and subscription_id <= len(self.contact_event_callbacks):
            self.contact_event_callbacks.pop(subscription_id - 1)

    def get_simulation_time_steps_per_second(self, stage_id: int, scene_path: int) -> int:
        """Get the mock simulation rate.

        Args:
            stage_id: Stage identifier, which does not affect the mock rate.
            scene_path: Encoded scene path, which does not affect the mock rate.

        Returns:
            Number of simulation steps per second.

        """
        del stage_id, scene_path
        return self.time_steps_per_second

    def get_simulation_timestamp(self) -> int:
        """Get the current mock timestamp.

        Returns:
            Number of completed mock steps.

        """
        return self.simulation_timestamp

    def get_simulation_step_count(self) -> int:
        """Get the current mock step count.

        Returns:
            Number of completed mock steps.

        """
        return self.simulation_step_count

    def subscribe_physics_on_step_events(self, pre_step: bool, order: int, on_update: StepCallback) -> int:
        """Register a physics-step callback.

        Args:
            pre_step: Whether the callback is requested before the step.
            order: Callback ordering key.
            on_update: Function notified for each step.

        Returns:
            One-based subscription identifier.

        """
        del pre_step, order
        self.step_event_callbacks.append(on_update)
        return len(self.step_event_callbacks)

    def unsubscribe_physics_on_step_events(self, subscription_id: int) -> None:
        """Remove a physics-step callback.

        Args:
            subscription_id: One-based identifier returned during subscription.

        """
        if subscription_id > 0 and subscription_id <= len(self.step_event_callbacks):
            self.step_event_callbacks.pop(subscription_id - 1)

    def is_capable_of_simulating(self, schema_names: list[str]) -> tuple[bool, list[bool]]:
        """Query mock support for applied schemas.

        Args:
            schema_names: Applied-schema names to query.

        Returns:
            Query status and one support flag per requested schema.

        """
        if not self.capability_check_enabled:
            return (False, [])
        capabilities = []
        for schema_name in schema_names:
            capabilities.append(self.supported_capabilities.get(schema_name, False))
        return (True, capabilities)

    def set_capability_check_enabled(self, enabled: bool) -> None:
        """Enable or disable successful capability queries.

        Args:
            enabled: Whether capability queries should succeed.

        """
        self.capability_check_enabled = enabled

    def set_supported_capability(self, schema_name: str, is_supported: bool) -> None:
        """Set mock support for one applied schema.

        Args:
            schema_name: Applied-schema name to configure.
            is_supported: Whether the mock reports support.

        """
        self.supported_capabilities[schema_name] = is_supported


def _setup_simulation_fns(simulation: Simulation, mock: MockSimulator) -> None:
    """Connect every simulation callback to a mock.

    Args:
        simulation: Simulation record whose callbacks are configured.
        mock: Callback implementation used by the record.

    """
    simulation.simulation_fns.initialize = mock.initialize
    simulation.simulation_fns.close = mock.close
    simulation.simulation_fns.get_attached_stage = mock.get_attached_stage
    simulation.simulation_fns.simulate_async = mock.simulate_async
    simulation.simulation_fns.simulate = mock.simulate
    simulation.simulation_fns.fetch_results = mock.fetch_results
    simulation.simulation_fns.check_results = mock.check_results
    simulation.simulation_fns.flush_changes = mock.flush_changes
    simulation.simulation_fns.pause_change_tracking = mock.pause_change_tracking
    simulation.simulation_fns.is_change_tracking_paused = mock.is_change_tracking_paused
    simulation.simulation_fns.subscribe_physics_contact_report_events = mock.subscribe_physics_contact_report_events
    simulation.simulation_fns.unsubscribe_physics_contact_report_events = mock.unsubscribe_physics_contact_report_events
    simulation.simulation_fns.get_simulation_time_steps_per_second = mock.get_simulation_time_steps_per_second
    simulation.simulation_fns.get_simulation_timestamp = mock.get_simulation_timestamp
    simulation.simulation_fns.get_simulation_step_count = mock.get_simulation_step_count
    simulation.simulation_fns.subscribe_physics_on_step_events = mock.subscribe_physics_on_step_events
    simulation.simulation_fns.unsubscribe_physics_on_step_events = mock.unsubscribe_physics_on_step_events
    simulation.simulation_fns.is_capable_of_simulating = mock.is_capable_of_simulating


class TestSimulatorSimulation:
    """Verify simulation lifecycle and callbacks through the manager."""

    def setup_method(self) -> None:
        """Register a fresh simulator mock before each test."""
        self.physics = physics_registration
        self.physics_simulation = physics_manager
        self.mock_simulator = MockSimulator()

        self.simulation = Simulation()
        _setup_simulation_fns(self.simulation, self.mock_simulator)

        self.simulation_id = self.physics.register_simulation(self.simulation, "MockSimulator")
        assert self.simulation_id != k_invalid_simulation_id
        self.mock_simulator.simulation_id = self.simulation_id

    def teardown_method(self) -> None:
        """Unregister the simulator mock after each test."""
        if self.simulation_id != k_invalid_simulation_id:
            self.physics.unregister_simulation(self.simulation_id)

    def test_stage_attachment(self) -> None:
        """Verify stage attachment, lookup, and close."""
        assert self.mock_simulator.attached_stage_id == 0
        assert self.physics_simulation.initialize(0, "123")
        assert self.mock_simulator.attached_stage_id == 123
        assert self.physics_simulation.get_attached_stage() == 123
        self.physics_simulation.close()
        assert self.mock_simulator.attached_stage_id == 0

    def test_simulation_timing(self) -> None:
        """Verify simulation rate, timestamp, and step progression."""
        assert self.physics_simulation.get_simulation_time_steps_per_second(self.simulation_id, 123, 0) == 60
        assert self.physics_simulation.get_simulation_timestamp(self.simulation_id) == 0
        assert self.physics_simulation.get_simulation_step_count(self.simulation_id) == 0

        self.physics_simulation.simulate_async(1.0 / 60.0, 0.0)
        self.physics_simulation.fetch_results()
        assert self.physics_simulation.get_simulation_timestamp(self.simulation_id) == 1

        for i in range(5):
            self.physics_simulation.simulate_async(1.0 / 60.0, (i + 1) * 1.0 / 60.0)
            self.physics_simulation.fetch_results()
        assert self.physics_simulation.get_simulation_timestamp(self.simulation_id) == 6

    def test_change_tracking(self) -> None:
        """Verify pausing and resuming change tracking."""
        assert not (self.physics_simulation.is_change_tracking_paused(self.simulation_id))
        self.physics_simulation.pause_change_tracking(True)
        assert self.physics_simulation.is_change_tracking_paused(self.simulation_id)
        self.physics_simulation.pause_change_tracking(False)
        assert not (self.physics_simulation.is_change_tracking_paused(self.simulation_id))

    def test_contact_callbacks(self) -> None:
        """Verify contact delivery while subscribed and silence after release."""
        contact_callback_called = False

        def on_contact_event(
            event_headers: ContactEventHeaderVector,
            contact_data: ContactDataVector,
            friction_anchors: FrictionAnchorsDataVector,
        ) -> None:
            nonlocal contact_callback_called
            del event_headers, contact_data, friction_anchors
            contact_callback_called = True

        sub = self.physics_simulation.subscribe_physics_contact_report_events(on_contact_event)
        assert sub is not None
        self.physics_simulation.simulate_async(1.0 / 60.0, 0.0)
        self.physics_simulation.fetch_results()
        assert contact_callback_called

        contact_callback_called = False
        sub = None
        self.physics_simulation.simulate_async(1.0 / 60.0, 1.0 / 60.0)
        self.physics_simulation.fetch_results()
        assert not (contact_callback_called)

    def test_step_callbacks(self) -> None:
        """Verify step elapsed time while subscribed and silence after release."""
        step_callback_called = False
        step_elapsed_time = None

        def on_step_event(elapsed_time: float, context: PhysicsStepContext) -> None:
            nonlocal step_callback_called, step_elapsed_time
            del context
            step_callback_called = True
            step_elapsed_time = elapsed_time

        sub = self.physics_simulation.subscribe_physics_on_step_events(False, 0, on_step_event)
        assert sub is not None

        elapsed_time = 1.0 / 60.0
        self.physics_simulation.simulate_async(elapsed_time, 0.0)
        self.physics_simulation.fetch_results()

        assert step_callback_called
        assert step_elapsed_time == pytest.approx(elapsed_time, rel=0.0, abs=1e-7)

        step_callback_called = False
        sub = None
        self.physics_simulation.simulate_async(1.0 / 60.0, 1.0 / 60.0)
        assert not (step_callback_called)

    def test_contact_event_type_enum(self) -> None:
        """Verify that contact event enum members are distinct."""
        assert ContactEventType.CONTACT_FOUND != ContactEventType.CONTACT_LOST
        assert ContactEventType.CONTACT_FOUND != ContactEventType.CONTACT_PERSIST

    def test_force_mode_enum(self) -> None:
        """Verify that force mode enum members are distinct."""
        assert ForceMode.FORCE != ForceMode.IMPULSE
        assert ForceMode.FORCE != ForceMode.VELOCITY_CHANGE
        assert ForceMode.FORCE != ForceMode.ACCELERATION

    def test_contact_event_header(self) -> None:
        """Verify contact event header field assignment and readback."""
        header = ContactEventHeader()
        header.type = ContactEventType.CONTACT_FOUND
        header.stage_id = 123
        header.actor0 = 456
        assert header.type == ContactEventType.CONTACT_FOUND
        assert header.stage_id == 123
        assert header.actor0 == 456

    def test_contact_data(self) -> None:
        """Verify contact position, normal, separation, and impulse fields."""
        contact = ContactData()
        contact.position = Float3(1.0, 2.0, 3.0)
        contact.normal = Float3(0.0, 1.0, 0.0)
        contact.separation = -0.1
        contact.impulse = Float3(5.0, 6.0, 7.0)
        assert (contact.position.x, contact.position.y, contact.position.z) == (1.0, 2.0, 3.0)
        assert contact.separation == pytest.approx(-0.1, rel=0.0, abs=1e-5)

    def test_contact_vectors(self) -> None:
        """Verify contact header vector construction and append."""
        header_vector = ContactEventHeaderVector()
        assert len(header_vector) == 0
        header1 = ContactEventHeader()
        header1.type = ContactEventType.CONTACT_FOUND
        header_vector.append(header1)
        assert len(header_vector) == 1

    def test_enhanced_contact_callbacks(self) -> None:
        """Verify that enhanced callbacks receive the native header vector."""
        received_headers = None

        def on_contact_event(
            event_headers: ContactEventHeaderVector,
            contact_data: ContactDataVector,
            friction_anchors: FrictionAnchorsDataVector,
        ) -> None:
            nonlocal received_headers
            del contact_data, friction_anchors
            received_headers = event_headers

        sub = self.physics_simulation.subscribe_physics_contact_report_events(on_contact_event)
        self.physics_simulation.simulate_async(1.0 / 60.0, 0.0)
        self.physics_simulation.fetch_results()
        assert isinstance(received_headers, ContactEventHeaderVector)
        sub = None


class TestSimulatorCapabilityCheck:
    """Verify per-simulation capability queries through the manager."""

    def setup_method(self) -> None:
        """Register a simulator mock with known capability flags."""
        self.physics = physics_registration
        self.physics_simulation = physics_manager
        self.mock_simulator = MockSimulator()
        self.mock_simulator.set_supported_capability("PhysicsRigidBodyAPI", True)
        self.mock_simulator.set_supported_capability("PhysicsCollisionAPI", True)
        self.mock_simulator.set_supported_capability("UnsupportedSchemaAPI", False)

        self.simulation = Simulation()
        _setup_simulation_fns(self.simulation, self.mock_simulator)

        self.simulation_id = self.physics.register_simulation(self.simulation, "MockCapabilitySim")
        assert self.simulation_id != k_invalid_simulation_id

    def teardown_method(self) -> None:
        """Unregister the capability mock after each test."""
        if self.simulation_id != k_invalid_simulation_id:
            self.physics.unregister_simulation(self.simulation_id)

    def test_check_single_supported_capability(self) -> None:
        """Verify a positive capability result."""
        success, caps = self.physics_simulation.is_capable_of_simulating(self.simulation_id, ["PhysicsRigidBodyAPI"])
        assert success
        assert caps[0]

    def test_check_single_unsupported_capability(self) -> None:
        """Verify a negative capability result from a successful query."""
        success, caps = self.physics_simulation.is_capable_of_simulating(self.simulation_id, ["UnsupportedSchemaAPI"])
        assert success
        assert not (caps[0])

    def test_check_capability_with_invalid_simulation_id(self) -> None:
        """Verify failure and an empty result for an invalid simulation identifier."""
        success, caps = self.physics_simulation.is_capable_of_simulating(
            k_invalid_simulation_id, ["PhysicsRigidBodyAPI"]
        )
        assert not (success)
        assert len(caps) == 0

    def test_check_empty_schema_list(self) -> None:
        """Verify a successful empty capability query."""
        success, caps = self.physics_simulation.is_capable_of_simulating(self.simulation_id, [])
        assert success
        assert len(caps) == 0

    def test_dynamic_capability_registration(self) -> None:
        """Verify that capability updates affect subsequent queries."""
        success, caps = self.physics_simulation.is_capable_of_simulating(self.simulation_id, ["NewDynamicAPI"])
        assert success
        assert not (caps[0])
        self.mock_simulator.set_supported_capability("NewDynamicAPI", True)
        success, caps = self.physics_simulation.is_capable_of_simulating(self.simulation_id, ["NewDynamicAPI"])
        assert success
        assert caps[0]
