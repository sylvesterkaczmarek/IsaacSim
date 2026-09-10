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

"""Validate Newton initialization from a caller-owned ovstage handle.

The initialization contract accepts a null ovstage as an empty scene, a
registered ovstage handle as the physics source, and rejects unregistered
handles or non-numeric USD identifiers without raising.
"""

from __future__ import annotations

import _physics_setup  # noqa: F401  -- puts the umbrella bindings on sys.path
import pytest

try:
    from isaacsim.physics_engines.ovnewton.impl.simulation_functions import NewtonSimulationFunctions
    from isaacsim.physics_engines.ovstage import get_native_handle, setup

    _NEWTON_AVAILABLE = True
except Exception:  # noqa: BLE001 -- newton stack (warp) may be absent in some envs
    _NEWTON_AVAILABLE = False


class _MockStage:
    """Record the arguments passed to a stand-in Newton stage."""

    def __init__(self) -> None:
        self.init_calls: list[tuple[object | None, int]] = []
        self._ovstage: object | None = None

    def initialize(self, ovstage_obj: object | None, stage_id: int = 0) -> bool:
        self.init_calls.append((ovstage_obj, stage_id))
        self._ovstage = ovstage_obj
        return True

    def has_attached_stage(self) -> bool:
        return self._ovstage is not None

    def close(self) -> None:
        self._ovstage = None


@pytest.mark.skipif(
    not _NEWTON_AVAILABLE,
    reason="isaacsim.physics_engines.ovnewton.impl not importable",
)
class TestNewtonInitializeContract:
    """Validate ovstage-handle handling for Newton initialize."""

    def setup_method(self) -> None:
        """Create a fresh recording stage and simulation-function adapter."""
        self.stage = _MockStage()
        self.fns = NewtonSimulationFunctions(self.stage)

    def test_non_numeric_id_fails_soft(self) -> None:
        """Verify that a non-numeric identifier fails without reaching the stage."""
        assert not (self.fns.initialize(0, "/World/some/path"))
        assert self.stage.init_calls == []

    def test_empty_ovstage_is_empty_scene(self) -> None:
        """Verify that a null ovstage initializes an empty scene."""
        assert self.fns.initialize(0, "")
        assert self.stage.init_calls == [(None, 0)]
        assert not self.fns.has_attached_stage()

    def test_unregistered_handle_fails_soft(self) -> None:
        """Verify that an unregistered native handle fails without raising."""
        assert not (self.fns.initialize(0xDEADBEEF, "0"))
        assert self.stage.init_calls == []
        assert not self.fns.has_attached_stage()

    def test_registered_ovstage_reaches_stage(self) -> None:
        """Verify that a registered ovstage handle and id reach the stage."""
        setup()
        import ovstage

        ov = ovstage.Stage("newton-init-contract")
        try:
            handle = get_native_handle(ov)
            assert self.fns.initialize(handle, "123")
            assert self.stage.init_calls == [(ov, 123)]
            assert self.fns.has_attached_stage()
        finally:
            ov.destroy()

    def test_nb_handle_capsule_reaches_stage(self) -> None:
        """Verify that a nanobind void* capsule is accepted like an int handle."""
        import ctypes

        from isaacsim.physics_engines.ovstage import as_native_handle

        setup()
        import ovstage

        ov = ovstage.Stage("newton-init-capsule")
        try:
            handle = get_native_handle(ov)
            ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
            ctypes.pythonapi.PyCapsule_New.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_void_p,
            ]
            capsule = ctypes.pythonapi.PyCapsule_New(handle, b"nb_handle", None)
            assert as_native_handle(capsule) == handle
            assert self.fns.initialize(capsule, "7")
            assert self.stage.init_calls == [(ov, 7)]
            assert self.fns.has_attached_stage()
        finally:
            ov.destroy()

    def test_failed_model_build_retains_ovstage_for_has_attached_stage(self) -> None:
        """Verify a failed build keeps the borrowed ovstage and reports attachment.

        The manager uses ``has_attached_stage`` (not ``get_attached_stage``) to
        classify ``eFailedDirty`` and pin the caller-owned owner after initialize
        returns false. Dropping that signal would let the caller free a stage
        Newton still holds for ``start_simulation`` retry.
        """
        from isaacsim.physics_engines.ovnewton.impl.newton_stage import NewtonStage

        newton = NewtonStage()
        borrowed = object()

        def _fail_build(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("forced model-build failure")

        newton._import_engine = lambda: (None, None, None)  # type: ignore[method-assign]
        newton._build_newton_model_from_ovstage = _fail_build  # type: ignore[method-assign]

        assert not newton.initialize(borrowed, stage_id=0)
        assert newton.has_attached_stage()
        assert newton._ovstage is borrowed
        assert not newton.initialized

        newton.close()
        assert not newton.has_attached_stage()
