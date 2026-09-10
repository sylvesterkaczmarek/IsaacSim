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

"""Verify attached-stage owner retention across success and failure paths.

The simulation holds the caller-owned ovstage's Python owner alive across an
attach so a caller can drop its own reference without freeing the borrowed
native instance under the engine. The correctness question is the *failure*
branch (a fallible detach retains the stage, so the owner must NOT be dropped) --
which the happy-path C++ attach test cannot reach.

The keepalive lives in the engine-agnostic core binding, so a mock backend whose
initialize()/close() return False on demand drives the binding's success flag
exactly like a real failed ovphysx detach -- no SDK fault injection needed. Owner
lifetime is observed through a weakref: still-alive means the keepalive holds it.
"""

from __future__ import annotations

import gc
import weakref

import _physics_setup  # noqa: F401  -- registers the ovphysx backend
import isaacsim.physics.manager as physics_manager
import isaacsim.physics.registration as physics_registration
from isaacsim.physics.manager.impl import (
    Simulation,
    k_invalid_simulation_id,
)


class _ControllableBackend:
    """Backend whose initialization and close outcomes are controlled by a test.

    Args:
        name: Backend name recorded in callback order entries.
        order: Optional list that receives initialization and close entries.

    """

    def __init__(self, name: str = "", order: list[tuple[str, str]] | None = None) -> None:
        self.name = name
        self._order = order
        self.initialize_result = True
        self.close_result = True
        self.init_raises = False
        self.close_raises = False
        self.attached_stage = 0
        self.has_attached_stage_result = False
        self.has_attached_stage_raises = False
        self.init_calls = 0
        self.close_calls = 0

    def initialize(self, ovstage: int, usd_identifier: str) -> bool:
        del ovstage, usd_identifier
        self.init_calls += 1
        if self._order is not None:
            self._order.append(("init", self.name))
        if self.init_raises:
            raise RuntimeError(f"init boom from {self.name}")
        return self.initialize_result

    def close(self) -> bool:
        self.close_calls += 1
        if self._order is not None:
            self._order.append(("close", self.name))
        if self.close_raises:
            raise RuntimeError(f"close boom from {self.name}")
        return self.close_result

    def get_attached_stage(self) -> int:
        return self.attached_stage

    def has_attached_stage(self) -> bool:
        if self.has_attached_stage_raises:
            raise RuntimeError(f"has_attached_stage boom from {self.name}")
        return self.has_attached_stage_result


class _Owner:
    """Sentinel stand-in for a caller-owned ovstage; tracked via weakref."""


class TestStageOwnerKeepalive:
    """Verify stage-owner lifetime across success and failure paths."""

    def setup_method(self) -> None:
        """Register a controllable backend before each test."""
        self.physics = physics_registration
        self.backend = _ControllableBackend()
        sim = Simulation()
        s = sim.simulation_fns
        s.initialize = self.backend.initialize
        s.close = self.backend.close
        s.get_attached_stage = self.backend.get_attached_stage
        s.has_attached_stage = self.backend.has_attached_stage
        self.simulation_id = self.physics.register_simulation(sim, "KeepaliveTest")
        assert self.simulation_id != k_invalid_simulation_id

    def teardown_method(self) -> None:
        """Close and unregister the controllable backend after each test."""
        # Force a clean keepalive slot regardless of what the test left behind.
        self.backend.close_result = True
        physics_manager.close()
        if self.simulation_id != k_invalid_simulation_id:
            self.physics.unregister_simulation(self.simulation_id)

    def _attach(self, ident: str) -> weakref.ref:
        """Attach with a fresh owner and retain only its weak reference.

        Args:
            ident: USD identifier supplied during initialization.

        Returns:
            Weak reference used to observe the owner's lifetime.

        """
        sim = physics_manager
        owner = _Owner()
        ref = weakref.ref(owner)
        assert sim.initialize(0, ident, owner=owner)
        del owner
        gc.collect()
        return ref

    def test_owner_kept_alive_while_attached(self) -> None:
        """Verify that attachment keeps the caller-owned stage owner alive."""
        ref = self._attach("1")
        assert ref() is not None, "keepalive must hold the owner while attached"

    def test_successful_close_releases_owner(self) -> None:
        """Verify that a successful close releases the attached owner."""
        ref = self._attach("1")
        assert physics_manager.close()
        gc.collect()
        assert ref() is None, "a successful close must release the owner"

    def test_successful_reinit_releases_previous_owner(self) -> None:
        """Verify that successful reinitialization replaces the previous owner."""
        ref1 = self._attach("1")
        self._attach("2")  # succeeds -> previous owner replaced
        assert ref1() is None, "a successful re-init must release the previous owner"

    def test_failed_reinit_retains_previous_owner(self) -> None:
        """Verify that failed reinitialization retains the active stage owner."""
        # A failed re-initialize leaves the previous stage attached, so its owner
        # must stay alive rather than be dropped under the engine.
        ref1 = self._attach("1")
        sim = physics_manager
        owner2 = _Owner()
        self.backend.initialize_result = False
        assert not (sim.initialize(0, "2", owner=owner2))
        del owner2
        gc.collect()
        assert ref1() is not None, "a failed re-init must retain the previous owner"

    def test_failed_close_retains_owner(self) -> None:
        """Verify that a failed close retains the still-attached owner."""
        # A failed close leaves the stage attached, so keep the owner.
        ref = self._attach("1")
        self.backend.close_result = False
        assert not (physics_manager.close())
        gc.collect()
        assert ref() is not None, "a failed close must retain the owner"

    def _register(self, backend: _ControllableBackend, name: str) -> int:
        """Register a controllable backend.

        Args:
            backend: Backend whose callbacks populate the simulation.
            name: Simulation name used for registration.

        Returns:
            Registered simulation identifier.

        """
        sim = Simulation()
        s = sim.simulation_fns
        s.initialize = backend.initialize
        s.close = backend.close
        s.get_attached_stage = backend.get_attached_stage
        s.has_attached_stage = backend.has_attached_stage
        sid = self.physics.register_simulation(sim, name)
        assert sid != k_invalid_simulation_id
        return sid

    def test_partial_init_failure_rolls_back_succeeded_backends(self) -> None:
        """Verify that partial initialization failure closes earlier successes."""
        # initialize() is broadcast to every backend, so a backend that
        # fails after another already succeeded (e.g. ovphysx attaches the ovstage,
        # then Newton fails) must roll back (close) the succeeded one -- otherwise the
        # aggregate returns false with a stage still attached and its owner unpinned,
        # a use-after-free once the caller drops it.
        #
        # The backend map iterates unordered, so drive the failure deterministically:
        # register several succeeding backends, learn the iteration order, then make a
        # backend that iterates *after* a known succeeder fail. That guarantees the
        # succeeder runs first and must therefore be rolled back.
        sim = physics_manager
        order: list[tuple[str, str]] = []
        backends = [_ControllableBackend(name=f"b{i}", order=order) for i in range(4)]
        ids = [self._register(b, f"OrderedBackend{i}") for i, b in enumerate(backends)]
        try:
            # Phase 1: learn the iteration order (all succeed).
            assert sim.initialize(0, "0")
            sim.close()
            ran = [name for kind, name in order if kind == "init" and name.startswith("b")]
            assert len(ran) >= 2, "need >=2 ordered backends to sequence"
            first_b = next(b for b in backends if b.name == ran[0])
            fail_b = next(b for b in backends if b.name == ran[1])

            # Phase 2: fail the second-iterated backend so the first succeeds before it.
            for b in backends:
                b.init_calls = 0
                b.close_calls = 0
            fail_b.initialize_result = False
            assert not (sim.initialize(0, "1"))
            assert first_b.init_calls >= 1, "the earlier backend should have initialized"
            assert (
                first_b.close_calls == first_b.init_calls
            ), "a backend that succeeded before a later failure must be rolled back (closed)"
        finally:
            for sid in ids:
                self.physics.unregister_simulation(sid)

    def test_rollback_survives_a_close_exception(self) -> None:
        """Verify that rollback continues after one backend close raises."""
        # rollback close() is best-effort. Newton's close() is Python and
        # can throw; a throwing close during rollback must not escape initialize() (it
        # would leak past the false return, or replace the original exception on the
        # rethrow path) nor stop the other succeeded backends from being closed.
        sim = physics_manager
        order: list[tuple[str, str]] = []
        backends = [_ControllableBackend(name=f"b{i}", order=order) for i in range(4)]
        ids = [self._register(b, f"OrderedBackend{i}") for i, b in enumerate(backends)]
        thrower = None
        try:
            # Phase 1: learn the iteration order.
            assert sim.initialize(0, "0")
            sim.close()
            ran = [name for kind, name in order if kind == "init" and name.startswith("b")]
            assert len(ran) >= 3, "need >=3 ordered backends to sequence"
            # ran[0] and ran[1] succeed; ran[2] fails init. Rollback closes in reverse,
            # so ran[1] is closed before ran[0]: make ran[1]'s close throw and assert
            # ran[0] is still closed afterwards.
            survivor = next(b for b in backends if b.name == ran[0])
            thrower = next(b for b in backends if b.name == ran[1])
            failer = next(b for b in backends if b.name == ran[2])

            for b in backends:
                b.init_calls = 0
                b.close_calls = 0
            thrower.close_raises = True
            failer.initialize_result = False

            # Must return False, not raise, despite thrower.close() throwing in rollback.
            assert not (sim.initialize(0, "1"))
            assert thrower.close_calls >= 1, "the throwing close was attempted"
            assert survivor.close_calls >= 1, "rollback must continue past a throwing close and close the others"
        finally:
            if thrower is not None:
                thrower.close_raises = False  # let any later close() run cleanly
            for sid in ids:
                self.physics.unregister_simulation(sid)

    def test_dirty_rollback_pins_the_new_owner(self) -> None:
        """Verify that an incomplete rollback retains the newly attached owner."""
        # If a succeeded backend's rollback close() cannot confirm the detach (returns
        # false), the new stage may still be attached. initialize()
        # then reports FailedDirty and the binding must pin the new owner defensively --
        # otherwise the caller drops it and the still-attached stage is freed under the
        # engine. Dropping the old owner is safe (its stage was already detached).
        sim = physics_manager
        order: list[tuple[str, str]] = []
        backends = [_ControllableBackend(name=f"b{i}", order=order) for i in range(4)]
        ids = [self._register(b, f"OrderedBackend{i}") for i, b in enumerate(backends)]
        try:
            assert sim.initialize(0, "0")
            sim.close()
            ran = [name for kind, name in order if kind == "init" and name.startswith("b")]
            assert len(ran) >= 2, "need >=2 ordered backends to sequence"
            dirty_b = next(b for b in backends if b.name == ran[0])  # succeeds; rollback close returns False
            fail_b = next(b for b in backends if b.name == ran[1])  # fails init

            for b in backends:
                b.init_calls = 0
                b.close_calls = 0
            dirty_b.close_result = False  # rollback close does not confirm the detach -> dirty
            fail_b.initialize_result = False

            owner = _Owner()
            ref = weakref.ref(owner)
            assert not (sim.initialize(0, "1", owner=owner))  # FailedDirty -> Python False
            del owner
            gc.collect()
            assert ref() is not None, "a dirty rollback must pin the new owner (stage may still be attached)"
        finally:
            for b in backends:
                b.close_result = True  # Let `teardown_method` close cleanly.
            for sid in ids:
                self.physics.unregister_simulation(sid)

    def test_throwing_backend_dirty_rollback_pins_the_new_owner(self) -> None:
        """Verify owner retention when initialization raises and rollback stays dirty."""
        # The throw path must be symmetric with the !ok path: a backend that *raises*
        # (rather than returns False) after another already attached is the same failure.
        # It must roll the succeeded backend back, must not propagate the exception (the
        # aggregate reports failure via its return), and when that rollback cannot confirm
        # the detach it must surface FailedDirty so the new owner is pinned -- otherwise the
        # exception escapes past the binding's pin and the still-attached stage is freed.
        sim = physics_manager
        order: list[tuple[str, str]] = []
        backends = [_ControllableBackend(name=f"b{i}", order=order) for i in range(4)]
        ids = [self._register(b, f"OrderedBackend{i}") for i, b in enumerate(backends)]
        try:
            assert sim.initialize(0, "0")
            sim.close()
            ran = [name for kind, name in order if kind == "init" and name.startswith("b")]
            assert len(ran) >= 2, "need >=2 ordered backends to sequence"
            dirty_b = next(b for b in backends if b.name == ran[0])  # succeeds; rollback close returns False
            raise_b = next(b for b in backends if b.name == ran[1])  # raises during init

            for b in backends:
                b.init_calls = 0
                b.close_calls = 0
            dirty_b.close_result = False  # rollback close does not confirm the detach -> dirty
            raise_b.init_raises = True

            owner = _Owner()
            ref = weakref.ref(owner)
            # A throwing backend converts to False (not propagated) and, dirty, pins the owner.
            assert not (sim.initialize(0, "1", owner=owner))
            del owner
            gc.collect()
            assert ref() is not None, "a throwing backend with a dirty rollback must pin the new owner"
        finally:
            for b in backends:
                b.init_raises = False
                b.close_result = True  # Let `teardown_method` close cleanly.
            for sid in ids:
                self.physics.unregister_simulation(sid)

    def test_reinit_detach_failure_retains_previous_owner(self) -> None:
        """Verify that failed old-stage detachment retains the previous owner."""
        # A re-initialize whose backend fails to detach the previous stage returns false
        # but leaves the OLD stage attached. The backend reports that via the explicit
        # has_attached_stage() signal -- crucially even when its StageCache id is 0 (a
        # caller-owned ovstage attached with an empty USD identifier), which
        # get_attached_stage() cannot tell apart from "nothing attached". The aggregate
        # must classify this as FailedDirty and the binding must retain the OLD owner;
        # dropping it would free a stage the backend still references. FailedDirty here
        # means "the old stage is attached", the opposite of a dirty rollback (new stage
        # attached), so the keepalive must keep both rather than swap to the new owner.
        sim = physics_manager
        # First init succeeds; the backend now holds a stage whose StageCache id is 0.
        self.backend.has_attached_stage_result = True
        self.backend.attached_stage = 0
        owner_a = _Owner()
        ref_a = weakref.ref(owner_a)
        assert sim.initialize(0, "1", owner=owner_a)
        del owner_a
        gc.collect()
        assert ref_a() is not None, "a successful init must keep its owner"

        # Re-init fails because the detach of the old stage failed; the old stage stays
        # attached (has_attached_stage() stays true) even though its id is 0.
        self.backend.initialize_result = False
        owner_b = _Owner()
        assert not (sim.initialize(0, "2", owner=owner_b))
        del owner_b
        gc.collect()
        assert ref_a() is not None, "a re-init that leaves the old stage attached must retain the old owner"
