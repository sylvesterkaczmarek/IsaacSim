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

"""Verify aggregation with Newton and OvPhysX registered concurrently."""

from __future__ import annotations

import _physics_setup  # noqa: F401  -- auto-registers ovphysx
import isaacsim.physics.manager as physics_manager
import isaacsim.physics.registration as physics_registration
import isaacsim.physics_engines.ovnewton.impl as newton_backend
import pytest
from common.usd_fixtures import freefall_stage, populate_ovstage
from isaacsim.physics.manager.impl import (
    Float3,
)


class TestMultiBackend:
    """Verify manager aggregation while Newton and OvPhysX are both active."""

    def setup_method(self) -> None:
        """Register Newton and resolve the process-wide OvPhysX backend."""
        self.physics = physics_registration
        self.newton_id = newton_backend.register()
        self.ovphysx_id = _physics_setup.find_ovphysx_sim_id()
        if self.ovphysx_id is None:
            pytest.skip("OvPhysX backend not registered")

    def teardown_method(self) -> None:
        """Unregister the Newton backend created for the test."""
        if self.newton_id is not None:
            newton_backend.unregister(self.newton_id)

    def test_both_backends_in_registry(self) -> None:
        """Verify that registry enumeration contains Newton and OvPhysX."""
        ids = self.physics.get_simulation_ids()
        assert self.newton_id in ids
        assert self.ovphysx_id in ids

    def test_both_backends_are_active(self) -> None:
        """Verify that Newton and OvPhysX are active after registration."""
        assert self.physics.is_simulation_active(self.newton_id)
        assert self.physics.is_simulation_active(self.ovphysx_id)

    def test_distinct_simulation_ids(self) -> None:
        """Verify that each backend receives a distinct simulation identifier."""
        assert self.newton_id != self.ovphysx_id

    def test_simulate_broadcasts_to_both(self) -> None:
        """Verify that one manager step advances both active backends."""
        sim = physics_manager
        sim.initialize(0, "123")
        before_n = sim.get_simulation_step_count(self.newton_id)
        before_o = sim.get_simulation_step_count(self.ovphysx_id)
        sim.simulate_async(1.0 / 60.0, 0.0)
        sim.fetch_results()
        after_n = sim.get_simulation_step_count(self.newton_id)
        after_o = sim.get_simulation_step_count(self.ovphysx_id)
        assert after_n >= before_n + 1
        assert after_o >= before_o + 1
        sim.close()

    def test_contact_subscription_does_not_raise_with_two_backends(self) -> None:
        """Verify contact subscription and simulation with two active backends."""
        sim = physics_manager
        stage_handle = freefall_stage()
        try:
            populate_ovstage(stage_handle)
            received = []
            subscription = sim.subscribe_physics_contact_report_events(lambda h, d, a: received.append(1))
            assert subscription is not None
            try:
                assert sim.initialize(
                    stage_handle.ovstage_handle,
                    str(stage_handle.stage_id),
                    owner=stage_handle.ovstage,
                )
                sim.simulate(1.0 / 60.0, 0.0)
            finally:
                subscription.unsubscribe()
                sim.close()
        finally:
            stage_handle.release()

    def test_raycast_does_not_raise_with_two_backends(self) -> None:
        """Verify that an aggregated closest raycast returns a boolean status."""
        sq = physics_manager
        result, hit = sq.raycast_closest(Float3(0.0, 0.0, 0.0), Float3(0.0, 0.0, 1.0), 100.0, False)
        assert isinstance(result, bool)

    def test_overlap_sums_across_backends(self) -> None:
        """Verify that aggregate overlap counts sum the two empty backend results."""
        sq = physics_manager
        count = sq.overlap_sphere(1.0, Float3(0.0, 0.0, 0.0), lambda hit: True)
        assert count == 0
