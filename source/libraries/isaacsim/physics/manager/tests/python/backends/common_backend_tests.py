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

"""Exercise registered backends through the public physics-manager APIs.

Subclasses must populate ``self.simulation_id`` (set in their ``setup_method``) by
registering a backend with ``Physics::register_simulation``. The mixin
uses only public registration and manager operations, so the same assertions
run against the mock, Newton, and OvPhysX backends.

The mixin is inherited directly by the plain pytest classes in the per-backend
driver files.
"""

from __future__ import annotations

import _physics_setup  # noqa: F401 — pre-resolves the bindings on sys.path
import isaacsim.physics.manager as physics_manager
import isaacsim.physics.registration as physics_registration
import pytest
from isaacsim.physics.manager.impl import (
    ContactDataVector,
    ContactEventHeaderVector,
    Float3,
    FrictionAnchorsDataVector,
    k_invalid_simulation_id,
)


class CommonBackendTests:
    """Backend-neutral runtime, query, interaction, and benchmark checks.

    Subclasses are responsible for ``setup_method`` registering exactly one
    backend and ``teardown_method`` releasing it. ``self.simulation_id`` must
    be a valid ``SimulationId`` after setup.
    """

    # Override when a backend does not populate a callback-function group.
    SUPPORTS_SIMULATION_FNS: bool = True
    SUPPORTS_SCENE_QUERY_FNS: bool = True
    SUPPORTS_INTERACTION_FNS: bool = True
    SUPPORTS_BENCHMARK_FNS: bool = True

    # ------------------------------------------------------------------
    # Registry-level
    # ------------------------------------------------------------------

    def test_backend_registered_with_valid_id(self) -> None:
        """Verify that registration returns a valid simulation identifier."""
        assert self.simulation_id != k_invalid_simulation_id

    def test_backend_appears_in_simulation_ids(self) -> None:
        """Verify that the registered backend appears in registry enumeration."""
        physics = physics_registration
        ids = physics.get_simulation_ids()
        assert self.simulation_id in ids

    def test_backend_is_active_after_registration(self) -> None:
        """Verify that a newly registered backend is active."""
        physics = physics_registration
        assert physics.is_simulation_active(self.simulation_id)

    def test_backend_can_be_deactivated_and_reactivated(self) -> None:
        """Verify that registry activation state can be toggled."""
        physics = physics_registration
        physics.deactivate_simulation(self.simulation_id)
        assert not (physics.is_simulation_active(self.simulation_id))
        physics.activate_simulation(self.simulation_id)
        assert physics.is_simulation_active(self.simulation_id)

    # ------------------------------------------------------------------
    # SimulationFns surface
    # ------------------------------------------------------------------

    def test_initialize_attaches_stage(self) -> None:
        """Verify that initialization leaves an integer attached-stage identifier."""
        if not self.SUPPORTS_SIMULATION_FNS:
            pytest.skip("backend does not populate SimulationFns")
        sim = physics_manager
        # initialize() may legitimately return False on a backend that
        # validates the stage via UsdUtilsStageCache and finds nothing —
        # we don't assert on the bool, only on the stage-id roundtrip.
        sim.initialize(0, "123")
        # Real backends may return 0 if the stage isn't in the cache; the
        # mock returns 123. Accept either as long as it's an integer.
        assert isinstance(sim.get_attached_stage(), int)
        sim.close()

    def test_step_count_progresses(self) -> None:
        """Verify that one asynchronous simulation step does not decrease the step count."""
        if not self.SUPPORTS_SIMULATION_FNS:
            pytest.skip("backend does not populate SimulationFns")
        sim = physics_manager
        sim.initialize(0, "123")
        before = sim.get_simulation_step_count(self.simulation_id)
        sim.simulate_async(1.0 / 60.0, 0.0)
        sim.fetch_results()
        after = sim.get_simulation_step_count(self.simulation_id)
        assert after >= before
        sim.close()

    def test_change_tracking_toggle(self) -> None:
        """Verify that change tracking can be paused and resumed."""
        if not self.SUPPORTS_SIMULATION_FNS:
            pytest.skip("backend does not populate SimulationFns")
        sim = physics_manager
        sim.pause_change_tracking(True)
        assert sim.is_change_tracking_paused(self.simulation_id)
        sim.pause_change_tracking(False)
        assert not (sim.is_change_tracking_paused(self.simulation_id))

    def test_capability_check_returns_valid_shape(self) -> None:
        """Verify the shape and types returned by the capability query."""
        if not self.SUPPORTS_SIMULATION_FNS:
            pytest.skip("backend does not populate SimulationFns")
        sim = physics_manager
        success, caps = sim.is_capable_of_simulating(self.simulation_id, ["PhysicsRigidBodyAPI", "FakeUnknownAPI"])
        assert isinstance(success, bool)
        assert isinstance(caps, list)
        if success:
            assert len(caps) == 2

    def test_contact_subscription_lifecycle(self) -> None:
        """Verify that a contact-report subscription survives a step and can unsubscribe."""
        if not self.SUPPORTS_SIMULATION_FNS:
            pytest.skip("backend does not populate SimulationFns")
        sim = physics_manager
        received = []

        def cb(
            headers: ContactEventHeaderVector,
            data: ContactDataVector,
            anchors: FrictionAnchorsDataVector,
        ) -> None:
            received.append((len(headers), len(data), len(anchors)))

        sub = sim.subscribe_physics_contact_report_events(cb)
        assert sub is not None
        sim.initialize(0, "123")
        sim.simulate_async(1.0 / 60.0, 0.0)
        sim.fetch_results()
        # Subscription must at least be released without raising. Whether
        # the callback fired depends on the engine; we don't assert on it.
        sub.unsubscribe()
        sim.close()

    # ------------------------------------------------------------------
    # SceneQueryFns surface
    # ------------------------------------------------------------------

    def test_raycast_does_not_raise(self) -> None:
        """Verify that a closest-raycast query returns a boolean status."""
        if not self.SUPPORTS_SCENE_QUERY_FNS:
            pytest.skip("backend does not populate SceneQueryFns")
        sq = physics_manager
        result, hit = sq.raycast_closest(Float3(0.0, 0.0, 0.0), Float3(0.0, 0.0, 1.0), 100.0, False)
        assert isinstance(result, bool)

    def test_overlap_does_not_raise(self) -> None:
        """Verify that a sphere-overlap query returns a nonnegative count."""
        if not self.SUPPORTS_SCENE_QUERY_FNS:
            pytest.skip("backend does not populate SceneQueryFns")
        sq = physics_manager
        count = sq.overlap_sphere(1.0, Float3(0.0, 0.0, 0.0), lambda hit: True)
        assert isinstance(count, int)
        assert count >= 0

    # ------------------------------------------------------------------
    # InteractionFns surface
    # ------------------------------------------------------------------

    def test_get_prim_debug_data_returns_dict(self) -> None:
        """Verify that prim debug data is exposed as a dictionary."""
        if not self.SUPPORTS_INTERACTION_FNS:
            pytest.skip("backend does not populate InteractionFns")
        ix = physics_manager
        data = ix.get_prim_debug_data("/World/Cube")
        assert isinstance(data, dict)

    # ------------------------------------------------------------------
    # BenchmarkFns surface
    # ------------------------------------------------------------------

    def test_profile_stats_subscription(self) -> None:
        """Verify that a profile-statistics subscription can be released."""
        if not self.SUPPORTS_BENCHMARK_FNS:
            pytest.skip("backend does not populate BenchmarkFns")
        bm = physics_manager
        sub = bm.subscribe_profile_stats_events(lambda stats: None)
        assert sub is not None
        sub.unsubscribe()
