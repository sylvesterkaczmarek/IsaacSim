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

"""Verify manager forwarding for raycast, sweep, and overlap scene queries."""

from __future__ import annotations

from collections.abc import Callable

import _physics_setup  # noqa: F401 - initializes Carbonite framework
import isaacsim.physics.manager as physics_manager
import isaacsim.physics.registration as physics_registration
from isaacsim.physics.manager.impl import (
    Float3,
    OverlapHit,
    RaycastHit,
    Simulation,
    SweepHit,
    k_invalid_simulation_id,
)

RaycastReport = Callable[[RaycastHit], bool]
SweepReport = Callable[[SweepHit], bool]
OverlapReport = Callable[[OverlapHit], bool]


class MockSceneQuery:
    """Test double that returns deterministic hits for supported scene queries."""

    def __init__(self) -> None:
        self.raycast_count = 0
        self.sweep_count = 0
        self.overlap_count = 0
        self.report_fn_calls = 0

    def raycast_closest(
        self,
        origin: Float3,
        unit_dir: Float3,
        distance: float,
        hit: RaycastHit,
        both_sides: bool,
    ) -> bool:
        """Record a closest-hit raycast and report success.

        Args:
            origin: Ray origin.
            unit_dir: Normalized ray direction.
            distance: Maximum ray distance.
            hit: Output hit populated by the manager wrapper.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            True to represent a hit.

        """
        del origin, unit_dir, distance, hit, both_sides
        self.raycast_count += 1
        return True

    def raycast_any(self, origin: Float3, unit_dir: Float3, distance: float, both_sides: bool) -> bool:
        """Record an any-hit raycast and report success.

        Args:
            origin: Ray origin.
            unit_dir: Normalized ray direction.
            distance: Maximum ray distance.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            True to represent a hit.

        """
        del origin, unit_dir, distance, both_sides
        self.raycast_count += 1
        return True

    def raycast_all(
        self,
        origin: Float3,
        unit_dir: Float3,
        distance: float,
        report_fn: RaycastReport | None,
        both_sides: bool,
    ) -> bool:
        """Record an all-hits raycast and publish one deterministic hit.

        Args:
            origin: Ray origin.
            unit_dir: Normalized ray direction.
            distance: Maximum ray distance.
            report_fn: Function notified with each hit, if supplied.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            True after completing the query.

        """
        del origin, unit_dir, distance, both_sides
        self.raycast_count += 1
        hit = RaycastHit()
        hit.position = Float3(1.0, 2.0, 3.0)
        hit.normal = Float3(0.0, 1.0, 0.0)
        hit.distance = 5.0
        if report_fn:
            self.report_fn_calls += 1
            report_fn(hit)
        return True

    def sweep_sphere_closest(
        self,
        radius: float,
        origin: Float3,
        unit_dir: Float3,
        distance: float,
        hit: SweepHit,
        both_sides: bool,
    ) -> bool:
        """Record a closest-hit sphere sweep and report success.

        Args:
            radius: Swept sphere radius.
            origin: Sweep origin.
            unit_dir: Normalized sweep direction.
            distance: Maximum sweep distance.
            hit: Output hit populated by the manager wrapper.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            True to represent a hit.

        """
        del radius, origin, unit_dir, distance, hit, both_sides
        self.sweep_count += 1
        return True

    def sweep_sphere_any(
        self,
        radius: float,
        origin: Float3,
        unit_dir: Float3,
        distance: float,
        both_sides: bool,
    ) -> bool:
        """Record an any-hit sphere sweep and report success.

        Args:
            radius: Swept sphere radius.
            origin: Sweep origin.
            unit_dir: Normalized sweep direction.
            distance: Maximum sweep distance.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            True to represent a hit.

        """
        del radius, origin, unit_dir, distance, both_sides
        self.sweep_count += 1
        return True

    def sweep_sphere_all(
        self,
        radius: float,
        origin: Float3,
        unit_dir: Float3,
        distance: float,
        report_fn: SweepReport | None,
        both_sides: bool,
    ) -> bool:
        """Record an all-hits sphere sweep and publish one deterministic hit.

        Args:
            radius: Swept sphere radius.
            origin: Sweep origin.
            unit_dir: Normalized sweep direction.
            distance: Maximum sweep distance.
            report_fn: Function notified with each hit, if supplied.
            both_sides: Whether both sides of mesh triangles are queried.

        Returns:
            True after completing the query.

        """
        del radius, origin, unit_dir, distance, both_sides
        self.sweep_count += 1
        hit = SweepHit()
        hit.position = Float3(1.0, 2.0, 3.0)
        hit.normal = Float3(0.0, 1.0, 0.0)
        hit.distance = 5.0
        if report_fn:
            self.report_fn_calls += 1
            report_fn(hit)
        return True

    def overlap_sphere(self, radius: float, pos: Float3, report_fn: OverlapReport | None) -> int:
        """Record a sphere overlap and publish one deterministic hit.

        Args:
            radius: Query sphere radius.
            pos: Query sphere position.
            report_fn: Function notified with each overlap, if supplied.

        Returns:
            One to represent the published overlap.

        """
        del radius, pos
        self.overlap_count += 1
        hit = OverlapHit()
        if report_fn:
            self.report_fn_calls += 1
            report_fn(hit)
        return 1

    def overlap_sphere_any(self, radius: float, pos: Float3) -> bool:
        """Record an any-hit sphere overlap and report success.

        Args:
            radius: Query sphere radius.
            pos: Query sphere position.

        Returns:
            True to represent an overlap.

        """
        del radius, pos
        self.overlap_count += 1
        return True


class TestSimulatorSceneQuery:
    """Verify scene-query forwarding through the physics manager."""

    def setup_method(self) -> None:
        """Register a fresh scene-query mock before each test."""
        self.physics = physics_registration
        self.physics_scene_query = physics_manager
        self.mock = MockSceneQuery()

        self.simulation = Simulation()
        self.simulation.scene_query_fns.raycast_closest = self.mock.raycast_closest
        self.simulation.scene_query_fns.raycast_any = self.mock.raycast_any
        self.simulation.scene_query_fns.raycast_all = self.mock.raycast_all
        self.simulation.scene_query_fns.sweep_sphere_closest = self.mock.sweep_sphere_closest
        self.simulation.scene_query_fns.sweep_sphere_any = self.mock.sweep_sphere_any
        self.simulation.scene_query_fns.sweep_sphere_all = self.mock.sweep_sphere_all
        self.simulation.scene_query_fns.overlap_sphere = self.mock.overlap_sphere
        self.simulation.scene_query_fns.overlap_sphere_any = self.mock.overlap_sphere_any

        self.simulation_id = self.physics.register_simulation(self.simulation, "MockSceneQuerySim")
        assert self.simulation_id != k_invalid_simulation_id

    def teardown_method(self) -> None:
        """Unregister the scene-query mock after each test."""
        if self.simulation_id != k_invalid_simulation_id:
            self.physics.unregister_simulation(self.simulation_id)

    def test_raycast_queries(self) -> None:
        """Verify closest, any-hit, and all-hits raycast forwarding."""
        origin = Float3(1.0, 2.0, 3.0)
        direction = Float3(0.0, 1.0, 0.0)

        result, hit = self.physics_scene_query.raycast_closest(origin, direction, 10.0, True)
        assert result
        assert self.mock.raycast_count == 1

        result = self.physics_scene_query.raycast_any(origin, direction, 10.0, True)
        assert result
        assert self.mock.raycast_count == 2

        hit_received = False

        def report_fn(hit: RaycastHit) -> bool:
            nonlocal hit_received
            hit_received = True
            assert isinstance(hit, RaycastHit)
            assert hit.distance == 5.0
            return True

        self.physics_scene_query.raycast_all(origin, direction, 10.0, report_fn, True)
        assert self.mock.raycast_count == 3
        assert hit_received

    def test_sweep_queries(self) -> None:
        """Verify closest, any-hit, and all-hits sphere-sweep forwarding."""
        origin = Float3(1.0, 2.0, 3.0)
        direction = Float3(0.0, 1.0, 0.0)

        result, hit = self.physics_scene_query.sweep_sphere_closest(1.0, origin, direction, 10.0, True)
        assert result
        assert self.mock.sweep_count == 1

        result = self.physics_scene_query.sweep_sphere_any(1.0, origin, direction, 10.0, True)
        assert result
        assert self.mock.sweep_count == 2

        hit_received = False

        def report_fn(hit: SweepHit) -> bool:
            nonlocal hit_received
            hit_received = True
            assert isinstance(hit, SweepHit)
            return True

        self.physics_scene_query.sweep_sphere_all(1.0, origin, direction, 10.0, report_fn, True)
        assert hit_received

    def test_overlap_queries(self) -> None:
        """Verify counted and any-hit sphere-overlap forwarding."""
        position = Float3(1.0, 2.0, 3.0)
        hit_received = False

        def report_fn(hit: OverlapHit) -> bool:
            nonlocal hit_received
            hit_received = True
            assert isinstance(hit, OverlapHit)
            return True

        result = self.physics_scene_query.overlap_sphere(1.0, position, report_fn)
        assert result == 1
        assert hit_received

        result = self.physics_scene_query.overlap_sphere_any(1.0, position)
        assert result
