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

"""Run CommonBackendTests against the in-tree Newton backend."""

from __future__ import annotations

import _physics_setup  # noqa: F401
import isaacsim.physics_engines.ovnewton.impl as newton_backend
from common.physics_tests import CommonPhysicsTests

from .common_backend_tests import CommonBackendTests


class TestNewtonBackend(CommonBackendTests):
    """Run the shared backend contract against Newton."""

    # Newton today populates only SimulationFns.
    SUPPORTS_SIMULATION_FNS = True
    SUPPORTS_SCENE_QUERY_FNS = False
    SUPPORTS_INTERACTION_FNS = False
    SUPPORTS_BENCHMARK_FNS = False

    def setup_method(self) -> None:
        """Register a fresh Newton backend before each test."""
        self.simulation_id = newton_backend.register()

    def teardown_method(self) -> None:
        """Unregister the Newton backend after each test."""
        if self.simulation_id is not None:
            newton_backend.unregister(self.simulation_id)


class TestNewtonPhysics(CommonPhysicsTests):
    """Physics-correctness tests for the Newton backend."""

    SUPPORTS_PHYSICS = True

    def setup_method(self) -> None:
        """Register Newton and obtain its stage registry before each test."""
        self.simulation_id = newton_backend.register()
        self._registry = newton_backend.get_registry(self.simulation_id)

    def teardown_method(self) -> None:
        """Unregister the Newton backend after each test."""
        if self.simulation_id is not None:
            newton_backend.unregister(self.simulation_id)

    def get_body_position(self, prim_path: str) -> tuple[float, float, float] | None:
        """Get a Newton rigid body's world-space position.

        Args:
            prim_path: Path of the rigid-body prim.

        Returns:
            Body position, or None when the Newton stage registry is unavailable.

        """
        if self._registry is None:
            return None
        return self._registry.stage.get_body_position(prim_path)
