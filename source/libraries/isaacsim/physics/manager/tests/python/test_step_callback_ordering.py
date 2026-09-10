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

"""Verify physics-step callback ordering and multi-engine context.

The Newton tests cover pre-step and post-step ordering around its solver. The
multi-engine tests verify that manager subscribers can distinguish Newton and
OvPhysX through each callback's simulation identifier.
"""

from __future__ import annotations

from collections.abc import Callable

import _physics_setup  # noqa: F401  -- auto-registers ovphysx
import isaacsim.physics.manager as physics_manager
import isaacsim.physics_engines.ovnewton.impl as newton_backend
import pytest
from isaacsim.physics.manager.impl import PhysicsStepContext

StepEvent = tuple[str, object]
StepCallback = Callable[[float, PhysicsStepContext], None]


def _make_newton_stage_with_mock_solver(*, solver_raises: bool = False) -> tuple[object, list[StepEvent]]:
    """Create an initialized Newton stage with an observable mock solver.

    Args:
        solver_raises: Whether the solver should raise while stepping.

    Returns:
        Newton stage and its ordered event log.

    """
    from isaacsim.physics_engines.ovnewton.impl import NewtonStage

    stage = NewtonStage()
    stage._initialized = True

    events: list[tuple[str, object]] = []

    class _MockSolver:
        @staticmethod
        def step(
            state_0: object,
            state_1: object,
            control: object,
            contacts: object,
            elapsed_time: float,
        ) -> None:
            del state_0, state_1, control, contacts
            events.append(("engine.step", elapsed_time))
            if solver_raises:
                raise RuntimeError("intentional solver failure")

        @staticmethod
        def update_contacts(contacts: object) -> None:
            del contacts
            events.append(("engine.update_contacts", None))

    class _MockModel:
        @staticmethod
        def collide(state: object, contacts: object) -> None:
            del state, contacts
            events.append(("engine.collide", None))

    class _MockState:
        def __init__(self, name: str) -> None:
            self.name = name
            self.clear_count = 0

        def clear_forces(self) -> None:
            self.clear_count += 1
            events.append((f"{self.name}.clear_forces", None))

    stage._solver = _MockSolver()
    stage._model = _MockModel()
    stage._state_0 = _MockState("state_0")
    stage._state_1 = _MockState("state_1")
    stage._control = None
    stage._contacts = None
    return stage, events


class _StepOrderingMixin:
    """Per-engine assertions for pre-/post-step subscriber ordering."""

    def _make_stage(self) -> tuple[object, list[StepEvent]]:
        """Create the backend stage and its ordered event log.

        Returns:
            Backend stage and mutable event log populated during simulation.

        """
        raise NotImplementedError

    def _engine_step_events(self) -> tuple[str, ...]:
        """Get event names that identify the backend solver step.

        Returns:
            Event names accepted as evidence of a solver step.

        """
        return ("engine.step",)

    def test_pre_step_fires_before_engine_step(self) -> None:
        """Verify that pre-step subscribers run before the backend solver."""
        stage, events = self._make_stage()

        def pre_callback(elapsed_time: float, ctx: PhysicsStepContext) -> None:
            del ctx
            events.append(("pre.A", elapsed_time))

        sub_id = stage.subscribe_step_event(pre_step=True, order=0, callback=pre_callback)
        try:
            stage.simulate_async(1.0 / 60.0, 0.0)
        finally:
            stage.unsubscribe_step_event(sub_id)

        names = [e[0] for e in events]
        pre_idx = names.index("pre.A")
        eng_idx = next(i for i, n in enumerate(names) if n in self._engine_step_events())
        assert pre_idx < eng_idx, f"pre-step subscriber fired after engine step; events={names}"

    def test_post_step_fires_after_engine_step(self) -> None:
        """Verify that post-step subscribers run after the backend solver."""
        stage, events = self._make_stage()

        def post_callback(elapsed_time: float, ctx: PhysicsStepContext) -> None:
            del ctx
            events.append(("post.A", elapsed_time))

        sub_id = stage.subscribe_step_event(pre_step=False, order=0, callback=post_callback)
        try:
            stage.simulate_async(1.0 / 60.0, 0.0)
        finally:
            stage.unsubscribe_step_event(sub_id)

        names = [e[0] for e in events]
        post_idx = names.index("post.A")
        eng_idx = next(i for i, n in enumerate(names) if n in self._engine_step_events())
        assert post_idx > eng_idx, f"post-step subscriber fired before engine step; events={names}"

    def test_within_phase_order_field_controls_sequence(self) -> None:
        """Verify ascending callback order within each step phase."""
        stage, events = self._make_stage()

        def make_cb(tag: str) -> StepCallback:
            return lambda elapsed_time, ctx: events.append((tag, elapsed_time))

        ids = [
            stage.subscribe_step_event(pre_step=True, order=20, callback=make_cb("pre.20")),
            stage.subscribe_step_event(pre_step=True, order=10, callback=make_cb("pre.10")),
            stage.subscribe_step_event(pre_step=True, order=30, callback=make_cb("pre.30")),
            stage.subscribe_step_event(pre_step=False, order=20, callback=make_cb("post.20")),
            stage.subscribe_step_event(pre_step=False, order=10, callback=make_cb("post.10")),
        ]
        try:
            stage.simulate_async(1.0 / 60.0, 0.0)
        finally:
            for s in ids:
                stage.unsubscribe_step_event(s)

        names = [e[0] for e in events if e[0].startswith("pre.") or e[0].startswith("post.")]
        assert names == ["pre.10", "pre.20", "pre.30", "post.10", "post.20"], f"unexpected subscriber ordering: {names}"

    def test_subscriber_exception_does_not_break_chain(self) -> None:
        """Verify that one subscriber exception does not suppress later callbacks."""
        stage, events = self._make_stage()

        def make_cb(tag: str) -> StepCallback:
            return lambda elapsed_time, ctx: events.append((tag, elapsed_time))

        def boom(elapsed_time: float, ctx: PhysicsStepContext) -> None:
            del ctx
            events.append(("boom", elapsed_time))
            raise RuntimeError("intentional")

        ids = [
            stage.subscribe_step_event(pre_step=True, order=10, callback=make_cb("pre.10")),
            stage.subscribe_step_event(pre_step=True, order=20, callback=boom),
            stage.subscribe_step_event(pre_step=True, order=30, callback=make_cb("pre.30")),
            stage.subscribe_step_event(pre_step=False, order=10, callback=boom),
            stage.subscribe_step_event(pre_step=False, order=20, callback=make_cb("post.20")),
        ]
        try:
            stage.simulate_async(1.0 / 60.0, 0.0)
        finally:
            for s in ids:
                stage.unsubscribe_step_event(s)

        names = [e[0] for e in events]
        assert names.count("boom") == 2
        assert "pre.10" in names
        assert "pre.30" in names
        assert "post.20" in names

    def test_unsubscribe_removes_callback(self) -> None:
        """Verify that an unsubscribed callback does not run."""
        stage, events = self._make_stage()

        def cb(elapsed_time: float, ctx: PhysicsStepContext) -> None:
            del ctx
            events.append(("cb", elapsed_time))

        sub_id = stage.subscribe_step_event(pre_step=True, order=0, callback=cb)
        stage.unsubscribe_step_event(sub_id)
        stage.simulate_async(1.0 / 60.0, 0.0)
        assert "cb" not in [e[0] for e in events]

    def test_no_subscribers_means_engine_step_still_runs(self) -> None:
        """Verify that the backend solver runs without subscribers."""
        stage, events = self._make_stage()
        stage.simulate_async(1.0 / 60.0, 0.0)
        names = [e[0] for e in events]
        assert any(n in self._engine_step_events() for n in names), f"engine step missing from events={names}"

    def test_context_carries_the_stages_simulation_id(self) -> None:
        """Verify that callback context carries the stage simulation identifier."""
        stage, events = self._make_stage()
        del events
        stage._simulation_id = 4242
        captured: list[PhysicsStepContext] = []

        sub_id = stage.subscribe_step_event(pre_step=True, order=0, callback=lambda et, ctx: captured.append(ctx))
        try:
            stage.simulate_async(1.0 / 60.0, 0.0)
        finally:
            stage.unsubscribe_step_event(sub_id)

        assert len(captured) == 1
        assert captured[0].simulation_id.id == 4242
        assert captured[0].scene_path == 0


class TestNewtonStepCallbackOrdering(_StepOrderingMixin):
    """Verify callback ordering and state transitions for Newton."""

    def _make_stage(self) -> tuple[object, list[StepEvent]]:
        """Create a Newton stage with a successful mock solver.

        Returns:
            Newton stage and mutable event log populated during simulation.

        """
        return _make_newton_stage_with_mock_solver()

    def test_successful_step_clears_new_current_state_after_engine_step(self) -> None:
        """Verify that a successful step swaps states before clearing new forces."""
        stage, events = _make_newton_stage_with_mock_solver()
        previous_state = stage._state_0
        next_state = stage._state_1

        stage.simulate_async(1.0 / 60.0, 0.0)

        assert stage._state_0 is next_state, "successful step did not swap Newton states"
        assert previous_state.clear_count == 0, f"successful step cleared the old state; events={events}"
        assert next_state.clear_count == 1, f"successful step retained one-shot forces; events={events}"
        names = [event[0] for event in events]
        assert names.index("engine.step") < names.index("state_1.clear_forces")

    def test_solver_exception_consumes_current_states_forces(self) -> None:
        """Verify that a failed solver step clears one-shot forces on current state."""
        stage, events = _make_newton_stage_with_mock_solver(solver_raises=True)
        current_state = stage._state_0

        stage.simulate_async(1.0 / 60.0, 0.0)

        assert stage._state_0 is current_state, "failed step unexpectedly swapped Newton states"
        assert current_state.clear_count == 1, f"failed step retained one-shot forces; events={events}"

    def test_failed_step_holds_counter_and_is_observable(self) -> None:
        """Verify that solver failure preserves counters and sets the failure flag."""
        stage, _events = _make_newton_stage_with_mock_solver(solver_raises=True)
        before_steps = stage.step_count
        before_timestamp = stage.timestamp

        stage.simulate_async(1.0 / 60.0, 0.0)

        assert stage.last_step_failed, "failed step was not observable via last_step_failed"
        assert stage.step_count == before_steps, "failed step advanced the step counter"
        assert stage.timestamp == before_timestamp, "failed step advanced the timestamp"

    def test_successful_step_advances_counter_and_clears_failure_flag(self) -> None:
        """Verify that success advances the counter and clears the failure flag."""
        stage, _events = _make_newton_stage_with_mock_solver()
        before_steps = stage.step_count

        stage.simulate_async(1.0 / 60.0, 0.0)

        assert not (stage.last_step_failed), "successful step reported a failure"
        assert stage.step_count == before_steps + 1, "successful step did not advance the step counter"


class TestNewtonSolverGating:
    """The DOF read-back sync is gated on the solver being maximal-coordinate."""

    def test_maximal_solver_classification(self) -> None:
        """Verify maximal-coordinate classification for every Newton solver family."""
        from unittest.mock import Mock

        import newton
        from isaacsim.physics_engines.ovnewton.impl.newton_stage import _is_maximal_solver

        for cls in (newton.solvers.SolverXPBD, newton.solvers.SolverSemiImplicit, newton.solvers.SolverVBD):
            assert _is_maximal_solver(Mock(spec=cls)), f"{cls.__name__} must classify as maximal"
        for cls in (newton.solvers.SolverMuJoCo, newton.solvers.SolverFeatherstone):
            assert not (_is_maximal_solver(Mock(spec=cls))), f"{cls.__name__} must not classify as maximal"

    def test_gate_runs_sync_only_for_maximal_solver(self) -> None:
        """Verify generalized-coordinate synchronization only for maximal solvers."""
        stage, _events = _make_newton_stage_with_mock_solver()
        calls: list[int] = []
        stage._sync_generalized_coordinates = lambda: calls.append(1)

        stage._solver_is_maximal = True
        stage.simulate_async(1.0 / 60.0, 0.0)
        assert len(calls) == 1, "maximal solver must run the post-step IK sync"

        stage._solver_is_maximal = False
        stage.simulate_async(1.0 / 60.0, 0.0)
        assert len(calls) == 1, "non-maximal solver must NOT run the IK sync"

    def test_sync_no_ops_without_articulations_or_model(self) -> None:
        """Verify synchronization safely exits without articulations or a model."""
        # model present, no articulations -> early-out; eval_ik is never reached, must not raise.
        stage, _events = _make_newton_stage_with_mock_solver()
        stage._sync_generalized_coordinates()
        stage._model = None
        stage._sync_generalized_coordinates()


class TestMultiEngineStepContext:
    """Verify step-event context while multiple backends are registered.

    A consumer subscribes through
    ``PhysicsSimulation.subscribe_physics_on_step_events`` and filters events
    by ``ctx.simulation_id``. The assertions cover the OvPhysX C++ adapter and
    the Newton Python stage.
    """

    def setup_method(self) -> None:
        """Resolve OvPhysX and register Newton before each test."""
        self.ovphysx_id = _physics_setup.find_ovphysx_sim_id()
        if self.ovphysx_id is None:
            pytest.skip("OvPhysX backend not registered")
        self.newton_id = newton_backend.register()

    def teardown_method(self) -> None:
        """Unregister Newton after each test."""
        if self.newton_id is not None:
            newton_backend.unregister(self.newton_id)
            self.newton_id = None

    def test_callback_distinguishes_engines_by_simulation_id(self) -> None:
        """Verify that aggregate callbacks expose both backend identifiers."""
        sim = physics_manager
        sim.initialize(0, "123")

        firings: list[int] = []
        sub = sim.subscribe_physics_on_step_events(False, 0, lambda elapsed, ctx: firings.append(ctx.simulation_id.id))
        try:
            sim.simulate_async(1.0 / 60.0, 0.0)
            sim.fetch_results()
        finally:
            sub.unsubscribe()
            sim.close()

        assert self.ovphysx_id.id in firings
        assert self.newton_id.id in firings

    def test_consumer_filters_by_simulation_id(self) -> None:
        """Verify that a subscriber can filter callbacks to OvPhysX."""
        sim = physics_manager
        sim.initialize(0, "123")

        ovphysx_only: list[float] = []

        def _cb(elapsed: float, ctx: PhysicsStepContext) -> None:
            if ctx.simulation_id.id == self.ovphysx_id.id:
                ovphysx_only.append(elapsed)

        sub = sim.subscribe_physics_on_step_events(False, 0, _cb)
        try:
            for _ in range(3):
                sim.simulate_async(1.0 / 60.0, 0.0)
                sim.fetch_results()
        finally:
            sub.unsubscribe()
            sim.close()

        assert len(ovphysx_only) == 3, "ovphysx callback should fire 3 times across 3 step pairs"
