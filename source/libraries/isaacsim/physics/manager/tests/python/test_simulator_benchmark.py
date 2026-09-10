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

"""Verify manager aggregation of physics profiling statistics."""

from __future__ import annotations

from collections.abc import Callable

import _physics_setup  # noqa: F401 - initializes Carbonite framework
import isaacsim.physics.manager as physics_manager
import isaacsim.physics.registration as physics_registration
import pytest
from isaacsim.physics.manager.impl import (
    PhysicsProfileStats,
    Simulation,
    k_invalid_simulation_id,
    k_invalid_subscription_id,
)

ProfileCallback = Callable[[list[PhysicsProfileStats]], None]


class MockBenchmark:
    """Test double that records and emits profile-statistics subscriptions."""

    def __init__(self) -> None:
        self.subscription_count = 0
        self.last_subscription_id = k_invalid_subscription_id
        self.profile_stats_callback: ProfileCallback | None = None

    def subscribe_profile_stats_events(self, on_event: ProfileCallback) -> int:
        """Register a profile-statistics callback.

        Args:
            on_event: Function notified with a batch of profile statistics.

        Returns:
            Subscription identifier assigned by the mock.

        """
        self.profile_stats_callback = on_event
        self.subscription_count += 1
        self.last_subscription_id = self.subscription_count
        return self.last_subscription_id

    def unsubscribe_profile_stats_events(self, subscription_id: int) -> None:
        """Remove the active callback when its identifier matches.

        Args:
            subscription_id: Identifier returned during subscription.

        """
        if subscription_id == self.last_subscription_id:
            self.profile_stats_callback = None
            self.last_subscription_id = k_invalid_subscription_id

    def simulate_profile_stats(self) -> None:
        """Emit a deterministic three-zone profile-statistics batch."""
        if self.profile_stats_callback:
            stats = [PhysicsProfileStats(), PhysicsProfileStats(), PhysicsProfileStats()]
            stats[0].zone_name = "Simulation"
            stats[0].ms = 16.6
            stats[1].zone_name = "Collision Detection"
            stats[1].ms = 5.2
            stats[2].zone_name = "Integration"
            stats[2].ms = 2.1
            self.profile_stats_callback(stats)

    def has_active_subscription(self) -> bool:
        """Check whether the mock currently holds a callback.

        Returns:
            True when a profile-statistics callback is registered.

        """
        return self.profile_stats_callback is not None


class TestSimulatorBenchmark:
    """Verify profile-statistics subscription behavior through the manager."""

    def setup_method(self) -> None:
        """Register a fresh benchmark mock before each test."""
        self.physics = physics_registration
        self.physics_benchmarks = physics_manager
        self.mock = MockBenchmark()

        self.simulation = Simulation()
        self.simulation.benchmark_fns.subscribe_profile_stats_events = self.mock.subscribe_profile_stats_events
        self.simulation.benchmark_fns.unsubscribe_profile_stats_events = self.mock.unsubscribe_profile_stats_events

        self.simulation_id = self.physics.register_simulation(self.simulation, "MockBenchmarkSim")
        assert self.simulation_id != k_invalid_simulation_id

    def teardown_method(self) -> None:
        """Unregister the benchmark mock after each test."""
        if self.simulation_id != k_invalid_simulation_id:
            self.physics.unregister_simulation(self.simulation_id)

    def test_profile_stats_subscription(self) -> None:
        """Verify delivery and values of a three-zone statistics batch."""
        callback_called = False
        received_stats: list[PhysicsProfileStats] = []

        def test_callback(stats: list[PhysicsProfileStats]) -> None:
            nonlocal callback_called, received_stats
            callback_called = True
            received_stats = stats

        sub = self.physics_benchmarks.subscribe_profile_stats_events(test_callback)
        assert sub != k_invalid_subscription_id
        assert self.mock.has_active_subscription()

        self.mock.simulate_profile_stats()

        assert callback_called
        assert len(received_stats) == 3
        assert received_stats[0].zone_name == "Simulation"
        assert received_stats[0].ms == pytest.approx(16.6, rel=0.0, abs=1e-5)

        sub = None
        assert not (self.mock.has_active_subscription())

    def test_no_callback_after_unsubscribe(self) -> None:
        """Verify that releasing a subscription prevents later delivery."""
        callback_called = False

        def test_callback(stats: list[PhysicsProfileStats]) -> None:
            nonlocal callback_called
            del stats
            callback_called = True

        sub = self.physics_benchmarks.subscribe_profile_stats_events(test_callback)
        sub = None
        assert not (self.mock.has_active_subscription())

        self.mock.simulate_profile_stats()
        assert not (callback_called)

    def test_callback_with_empty_stats(self) -> None:
        """Verify that subscribers receive an empty statistics batch."""
        callback_called = False
        received_stats: list[PhysicsProfileStats] = []

        def test_callback(stats: list[PhysicsProfileStats]) -> None:
            nonlocal callback_called, received_stats
            callback_called = True
            received_stats = stats

        sub = self.physics_benchmarks.subscribe_profile_stats_events(test_callback)
        if self.mock.profile_stats_callback:
            self.mock.profile_stats_callback([])
        assert callback_called
        assert len(received_stats) == 0
        sub = None
