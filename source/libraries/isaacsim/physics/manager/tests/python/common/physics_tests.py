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

"""Verify physical state evolution across registered backends.

Each test builds a USD stage with rigid bodies, registers a backend, advances
the simulation through ``PhysicsSimulation::simulate_async``, and reads the
backend state to confirm that the bodies moved.

Subclasses must implement :meth:`get_body_position(prim_path)` —
backends differ in how they expose engine state (Newton via
``state_0.body_q`` and OvPhysX through its tensor binding).
"""

from __future__ import annotations

from types import ModuleType

import _physics_setup  # noqa: F401
import isaacsim.physics.manager as physics_manager
import pytest

from .usd_fixtures import StageHandle, freefall_stage, populate_ovstage


class CommonPhysicsTests:
    """Mixin: physics-correctness tests every backend must pass.

    Subclasses must:

    1. ``setup_method`` registers exactly one backend and stores its
       :class:`SimulationId` on ``self.simulation_id``, and prepare the
       backend so :meth:`get_body_position` returns a sensible answer
       for any prim path the backend was given via ``initialize``.
    2. ``teardown_method`` unregisters that backend.
    3. Implement :meth:`get_body_position(prim_path)` to return the
       latest ``(x, y, z)`` for the body matching ``prim_path``, or
       ``None`` if the backend doesn't know about it.
    """

    SUPPORTS_PHYSICS: bool = True
    """Set to ``False`` in subclasses whose engine integration isn't
    wired yet — tests will skip rather than fail."""

    def get_body_position(self, prim_path: str) -> tuple[float, float, float] | None:
        """Get the latest world-space position for a rigid body.

        Args:
            prim_path: Path of the rigid-body prim.

        Returns:
            Body position, or None when the backend has no matching body.

        """
        raise NotImplementedError("Per-backend driver must implement get_body_position()")

    def initialize_for_stage(self, sim: ModuleType, stage_handle: StageHandle) -> None:
        """Initialize a backend from a populated ovstage.

        Args:
            sim: Physics manager module used for initialization.
            stage_handle: Cached stage whose on-disk USD is populated into ovstage.

        """
        populate_ovstage(stage_handle)
        # A failed attach leaves physics a no-op, which surfaces later as a confusing
        # "body never moved" assertion; fail here instead.
        assert sim.initialize(
            stage_handle.ovstage_handle,
            str(stage_handle.stage_id),
            owner=stage_handle.ovstage,
        ), "physics_manager.initialize() failed for the populated ovstage"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _step(self, sim: ModuleType, dt: float, n: int) -> None:
        """Advance a simulation and fetch the final results.

        Args:
            sim: Physics manager module used for stepping.
            dt: Time interval per step in seconds.
            n: Number of simulation steps.

        """
        for _ in range(n):
            sim.simulate_async(dt, 0.0)
        sim.fetch_results()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_freefall_under_gravity(self) -> None:
        """Verify that a free cube falls a physically plausible distance.

        A cube starting at ``y=10`` without a ground plane should lose between
        three and seven meters of altitude after one simulated second.
        """
        if not self.SUPPORTS_PHYSICS:
            pytest.skip("backend physics integration not wired yet")

        stage_handle = freefall_stage(initial_height=10.0)
        try:
            sim = physics_manager
            self.initialize_for_stage(sim, stage_handle)
            try:
                # Confirm the backend parsed the cube before we ask it
                # to simulate.
                initial = self.get_body_position("/World/Cube")
                assert initial is not None, "backend did not register the cube; check USD parsing"
                assert initial[1] == pytest.approx(10.0, rel=0.0, abs=10 ** -(2))

                # 1 second of simulation at 60 Hz.
                self._step(sim, 1.0 / 60.0, 60)

                final = self.get_body_position("/World/Cube")
                assert final is not None
                # Should have fallen at least 3 m — generous bound that
                # covers solver substep / iteration variations across
                # backends but rules out "didn't move at all".
                drop = initial[1] - final[1]
                assert drop > 3.0, (
                    f"cube didn't fall under gravity: y went from "
                    f"{initial[1]:.3f} to {final[1]:.3f} (drop={drop:.3f})"
                )
                # And not more than the analytic free-fall, which is
                # 4.905 m for g=9.81 / t=1.0. Allow some over-shoot for
                # finite step error.
                assert drop < 7.0, f"cube fell unrealistically far: {drop:.3f} m in 1 s"
            finally:
                sim.close()
        finally:
            stage_handle.release()
