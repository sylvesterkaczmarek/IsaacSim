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

"""Exercise the shared backend contract with a fully populated mock.

The mock implements every callback-function group, allowing the common suite
to validate the complete public physics-manager surface.
"""

from __future__ import annotations

from collections.abc import Callable

import _physics_setup  # noqa: F401
import isaacsim.physics.registration as physics_registration
from isaacsim.physics.manager.impl import (
    ContactDataVector,
    ContactEventHeaderVector,
    DebugDataItemType,
    Float3,
    Float4,
    FrictionAnchorsDataVector,
    PhysicsStepContext,
    Simulation,
    k_invalid_simulation_id,
)

from .common_backend_tests import CommonBackendTests

ContactCallback = Callable[[ContactEventHeaderVector, ContactDataVector, FrictionAnchorsDataVector], None]
StepCallback = Callable[[float, PhysicsStepContext], None]
ProfileCallback = Callable[[object], None]
HitCallback = Callable[[object], bool]


class MockBackend:
    """Observable test double that implements every physics callback group."""

    def __init__(self) -> None:
        self.attached_stage_id = 0
        self.step_count = 0
        self.timestamp = 0
        self.change_tracking_paused = False
        self.contact_subs: dict[int, ContactCallback] = {}
        self.step_subs: dict[int, StepCallback] = {}
        self.profile_subs: dict[int, ProfileCallback] = {}
        self._next_sub = 1

    # SimulationFns
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
        """Get the currently attached stage identifier.

        Returns:
            Attached stage identifier, or zero when closed.

        """
        return self.attached_stage_id

    def simulate_async(self, e: float, c: float) -> None:
        """Record one step and notify step subscribers.

        Args:
            e: Simulated time interval in seconds.
            c: Simulation time before the step.

        """
        del c
        self.step_count += 1
        self.timestamp += 1
        ctx = PhysicsStepContext()
        for cb in self.step_subs.values():
            cb(e, ctx)

    def simulate(self, e: float, c: float) -> None:
        """Record one synchronous step and publish its results.

        Args:
            e: Simulated time interval in seconds.
            c: Simulation time before the step.

        """
        self.simulate_async(e, c)
        self.fetch_results()

    def fetch_results(self) -> None:
        """Publish empty contact vectors to every contact subscriber."""
        h = ContactEventHeaderVector()
        d = ContactDataVector()
        a = FrictionAnchorsDataVector()
        for cb in self.contact_subs.values():
            cb(h, d, a)

    def check_results(self) -> bool:
        """Report that the mock has valid results.

        Returns:
            True because result validation cannot fail in the mock.

        """
        return True

    def flush_changes(self) -> None:
        """Accept a request to flush pending changes."""

    def pause_change_tracking(self, p: bool) -> None:
        """Set the mock change-tracking state.

        Args:
            p: Whether change tracking should be paused.

        """
        self.change_tracking_paused = bool(p)

    def is_change_tracking_paused(self) -> bool:
        """Check whether change tracking is paused.

        Returns:
            Current change-tracking state.

        """
        return self.change_tracking_paused

    def subscribe_physics_contact_report_events(self, cb: ContactCallback) -> int:
        """Register a contact-report callback.

        Args:
            cb: Function notified with contact event vectors.

        Returns:
            Subscription identifier assigned by the mock.

        """
        sid = self._next_sub
        self._next_sub += 1
        self.contact_subs[sid] = cb
        return sid

    def unsubscribe_physics_contact_report_events(self, sid: int) -> None:
        """Remove a contact-report callback.

        Args:
            sid: Identifier returned during subscription.

        """
        self.contact_subs.pop(sid, None)

    def get_simulation_time_steps_per_second(self, stage_id: int, scene_path: str) -> int:
        """Get the mock simulation rate.

        Args:
            stage_id: Stage identifier, which does not affect the mock rate.
            scene_path: Physics scene path, which does not affect the mock rate.

        Returns:
            Number of simulation steps per second.

        """
        del stage_id, scene_path
        return 60

    def get_simulation_timestamp(self) -> int:
        """Get the number of timestamps recorded by the mock.

        Returns:
            Current mock timestamp.

        """
        return self.timestamp

    def get_simulation_step_count(self) -> int:
        """Get the number of simulation steps recorded by the mock.

        Returns:
            Current simulation step count.

        """
        return self.step_count

    def subscribe_physics_on_step_events(self, pre: bool, order: int, cb: StepCallback) -> int:
        """Register a physics-step callback.

        Args:
            pre: Whether the callback is requested before the step.
            order: Callback ordering key.
            cb: Function notified for each step.

        Returns:
            Subscription identifier assigned by the mock.

        """
        del pre, order
        sid = self._next_sub
        self._next_sub += 1
        self.step_subs[sid] = cb
        return sid

    def unsubscribe_physics_on_step_events(self, sid: int) -> None:
        """Remove a physics-step callback.

        Args:
            sid: Identifier returned during subscription.

        """
        self.step_subs.pop(sid, None)

    def is_capable_of_simulating(self, names: list[str]) -> tuple[bool, list[bool]]:
        """Report every requested schema as supported.

        Args:
            names: Applied-schema names to query.

        Returns:
            Successful-query status and one true capability flag per schema.

        """
        return (True, [True for _ in names])

    def handle_raycast(self, o: Float3, d: Float3, i: bool) -> None:
        """Accept an interactive raycast request.

        Args:
            o: Ray origin.
            d: Ray direction.
            i: Whether the interaction input is active.

        """
        del o, d, i

    # SceneQueryFns — minimal "no hit" implementations.
    # `closest` variants take a hit out-parameter (5th / 7th arg).
    def raycast_closest(
        self,
        origin: Float3,
        unit_dir: Float3,
        distance: float,
        hit: object,
        both_sides: bool,
    ) -> bool:
        """Report no closest hit for a raycast.

        Args:
            origin: Ray origin.
            unit_dir: Normalized ray direction.
            distance: Maximum ray distance.
            hit: Output object that would receive hit data.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            False because the mock scene is empty.

        """
        del origin, unit_dir, distance, hit, both_sides
        return False

    def raycast_any(self, origin: Float3, unit_dir: Float3, distance: float, both_sides: bool) -> bool:
        """Report no hit for an any-hit raycast.

        Args:
            origin: Ray origin.
            unit_dir: Normalized ray direction.
            distance: Maximum ray distance.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            False because the mock scene is empty.

        """
        del origin, unit_dir, distance, both_sides
        return False

    def raycast_all(
        self,
        origin: Float3,
        unit_dir: Float3,
        distance: float,
        report_fn: HitCallback,
        both_sides: bool,
    ) -> None:
        """Complete an all-hits raycast without reporting hits.

        Args:
            origin: Ray origin.
            unit_dir: Normalized ray direction.
            distance: Maximum ray distance.
            report_fn: Function that would receive each hit.
            both_sides: Whether both sides of mesh triangles are queried.

        """
        del origin, unit_dir, distance, report_fn, both_sides

    def sweep_sphere_closest(
        self,
        radius: float,
        origin: Float3,
        unit_dir: Float3,
        distance: float,
        hit: object,
        both_sides: bool,
    ) -> bool:
        """Report no closest hit for a sphere sweep.

        Args:
            radius: Swept sphere radius.
            origin: Sweep origin.
            unit_dir: Normalized sweep direction.
            distance: Maximum sweep distance.
            hit: Output object that would receive hit data.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            False because the mock scene is empty.

        """
        del radius, origin, unit_dir, distance, hit, both_sides
        return False

    def sweep_sphere_any(self, *a: object) -> bool:
        """Report no hit for an any-hit sphere sweep.

        Args:
            *a: Sphere-sweep arguments supplied by the native callback wrapper.

        Returns:
            False because the mock scene is empty.

        """
        del a
        return False

    def sweep_sphere_all(self, *a: object) -> None:
        """Complete an all-hits sphere sweep without reporting hits.

        Args:
            *a: Sphere-sweep arguments supplied by the native callback wrapper.

        """
        del a

    def sweep_box_closest(
        self,
        half_extent: Float3,
        position: Float3,
        rotation: Float4,
        unit_dir: Float3,
        distance: float,
        hit: object,
        both_sides: bool,
    ) -> bool:
        """Report no closest hit for a box sweep.

        Args:
            half_extent: Box half extents.
            position: Initial box position.
            rotation: Initial box orientation.
            unit_dir: Normalized sweep direction.
            distance: Maximum sweep distance.
            hit: Output object that would receive hit data.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            False because the mock scene is empty.

        """
        del half_extent, position, rotation, unit_dir, distance, hit, both_sides
        return False

    def sweep_box_any(self, *a: object) -> bool:
        """Report no hit for an any-hit box sweep.

        Args:
            *a: Box-sweep arguments supplied by the native callback wrapper.

        Returns:
            False because the mock scene is empty.

        """
        del a
        return False

    def sweep_box_all(self, *a: object) -> None:
        """Complete an all-hits box sweep without reporting hits.

        Args:
            *a: Box-sweep arguments supplied by the native callback wrapper.

        """
        del a

    def sweep_shape_closest(
        self,
        g_prim_path: int,
        unit_dir: Float3,
        distance: float,
        hit: object,
        both_sides: bool,
    ) -> bool:
        """Report no closest hit for a geometry-prim sweep.

        Args:
            g_prim_path: Encoded path of the swept geometry prim.
            unit_dir: Normalized sweep direction.
            distance: Maximum sweep distance.
            hit: Output object that would receive hit data.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            False because the mock scene is empty.

        """
        del g_prim_path, unit_dir, distance, hit, both_sides
        return False

    def sweep_shape_any(self, *a: object) -> bool:
        """Report no hit for an any-hit geometry-prim sweep.

        Args:
            *a: Geometry-prim sweep arguments supplied by the native callback wrapper.

        Returns:
            False because the mock scene is empty.

        """
        del a
        return False

    def sweep_shape_all(self, *a: object) -> None:
        """Complete an all-hits geometry-prim sweep without reporting hits.

        Args:
            *a: Geometry-prim sweep arguments supplied by the native callback wrapper.

        """
        del a

    def overlap_sphere(self, r: float, p: Float3, fn: HitCallback) -> int:
        """Report no overlaps for a sphere.

        Args:
            r: Query sphere radius.
            p: Query sphere position.
            fn: Function that would receive each overlap.

        Returns:
            Zero because the mock scene is empty.

        """
        del r, p, fn
        return 0

    def overlap_sphere_any(self, r: float, p: Float3) -> bool:
        """Report no overlap for an any-hit sphere query.

        Args:
            r: Query sphere radius.
            p: Query sphere position.

        Returns:
            False because the mock scene is empty.

        """
        del r, p
        return False

    def overlap_box(
        self,
        h: Float3,
        p: Float3,
        r: Float4,
        fn: HitCallback,
    ) -> int:
        """Report no overlaps for a box.

        Args:
            h: Query box half extents.
            p: Query box position.
            r: Query box orientation.
            fn: Function that would receive each overlap.

        Returns:
            Zero because the mock scene is empty.

        """
        del h, p, r, fn
        return 0

    def overlap_box_any(self, h: Float3, p: Float3, r: Float4) -> bool:
        """Report no overlap for an any-hit box query.

        Args:
            h: Query box half extents.
            p: Query box position.
            r: Query box orientation.

        Returns:
            False because the mock scene is empty.

        """
        del h, p, r
        return False

    def overlap_shape(self, g: int, fn: HitCallback) -> int:
        """Report no overlaps for a geometry prim.

        Args:
            g: Encoded path of the query geometry prim.
            fn: Function that would receive each overlap.

        Returns:
            Zero because the mock scene is empty.

        """
        del g, fn
        return 0

    def overlap_shape_any(self, g: int) -> bool:
        """Report no overlap for an any-hit geometry-prim query.

        Args:
            g: Encoded path of the query geometry prim.

        Returns:
            False because the mock scene is empty.

        """
        del g
        return False

    # InteractionFns
    def get_prim_debug_data(self, prim_path: str) -> dict[str, dict[str, object]]:
        """Get deterministic debug data for a prim.

        Args:
            prim_path: Prim path requested by the manager.

        Returns:
            One string-valued debug entry for the mock.

        """
        del prim_path
        return {"mock_key": {"type": DebugDataItemType.STRING, "value": "mock"}}

    # BenchmarkFns
    def subscribe_profile_stats_events(self, cb: ProfileCallback) -> int:
        """Register a profile-statistics callback.

        Args:
            cb: Function notified with profile statistics.

        Returns:
            Subscription identifier assigned by the mock.

        """
        sid = self._next_sub
        self._next_sub += 1
        self.profile_subs[sid] = cb
        return sid

    def unsubscribe_profile_stats_events(self, sid: int) -> None:
        """Remove a profile-statistics callback.

        Args:
            sid: Identifier returned during subscription.

        """
        self.profile_subs.pop(sid, None)


def _build_simulation(mock: MockBackend) -> Simulation:
    sim = Simulation()

    s = sim.simulation_fns
    s.initialize = mock.initialize
    s.close = mock.close
    s.get_attached_stage = mock.get_attached_stage
    s.simulate_async = mock.simulate_async
    s.simulate = mock.simulate
    s.fetch_results = mock.fetch_results
    s.check_results = mock.check_results
    s.flush_changes = mock.flush_changes
    s.pause_change_tracking = mock.pause_change_tracking
    s.is_change_tracking_paused = mock.is_change_tracking_paused
    s.subscribe_physics_contact_report_events = mock.subscribe_physics_contact_report_events
    s.unsubscribe_physics_contact_report_events = mock.unsubscribe_physics_contact_report_events
    s.get_simulation_time_steps_per_second = mock.get_simulation_time_steps_per_second
    s.get_simulation_timestamp = mock.get_simulation_timestamp
    s.get_simulation_step_count = mock.get_simulation_step_count
    s.subscribe_physics_on_step_events = mock.subscribe_physics_on_step_events
    s.unsubscribe_physics_on_step_events = mock.unsubscribe_physics_on_step_events
    s.is_capable_of_simulating = mock.is_capable_of_simulating

    q = sim.scene_query_fns
    q.raycast_closest = mock.raycast_closest
    q.raycast_any = mock.raycast_any
    q.raycast_all = mock.raycast_all
    q.sweep_sphere_closest = mock.sweep_sphere_closest
    q.sweep_sphere_any = mock.sweep_sphere_any
    q.sweep_sphere_all = mock.sweep_sphere_all
    q.sweep_box_closest = mock.sweep_box_closest
    q.sweep_box_any = mock.sweep_box_any
    q.sweep_box_all = mock.sweep_box_all
    q.sweep_shape_closest = mock.sweep_shape_closest
    q.sweep_shape_any = mock.sweep_shape_any
    q.sweep_shape_all = mock.sweep_shape_all
    q.overlap_sphere = mock.overlap_sphere
    q.overlap_sphere_any = mock.overlap_sphere_any
    q.overlap_box = mock.overlap_box
    q.overlap_box_any = mock.overlap_box_any
    q.overlap_shape = mock.overlap_shape
    q.overlap_shape_any = mock.overlap_shape_any

    i = sim.interaction_fns
    i.handle_raycast = mock.handle_raycast
    i.get_prim_debug_data = mock.get_prim_debug_data

    b = sim.benchmark_fns
    b.subscribe_profile_stats_events = mock.subscribe_profile_stats_events
    b.unsubscribe_profile_stats_events = mock.unsubscribe_profile_stats_events

    return sim


class TestMockBackend(CommonBackendTests):
    """Run the shared backend contract against the fully populated mock."""

    def setup_method(self) -> None:
        """Register a fresh mock backend before each test."""
        self.physics = physics_registration
        self.mock = MockBackend()
        self.simulation = _build_simulation(self.mock)
        self.simulation_id = self.physics.register_simulation(self.simulation, "MockBackend")
        assert self.simulation_id != k_invalid_simulation_id

    def teardown_method(self) -> None:
        """Unregister the mock backend after each test."""
        if self.simulation_id != k_invalid_simulation_id:
            self.physics.unregister_simulation(self.simulation_id)
