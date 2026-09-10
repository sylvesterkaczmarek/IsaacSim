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

"""Run CommonBackendTests against the ovphysx C++ backend.

The ovphysx backend is registered by _physics_setup.py at import time.
"""

from __future__ import annotations

import _physics_setup as _ps  # noqa: F401  -- registers the ovphysx backend
import pytest

from .common_backend_tests import CommonBackendTests


class TestOvPhysxBackend(CommonBackendTests):
    """Run the shared backend contract against process-wide OvPhysX."""

    SUPPORTS_SIMULATION_FNS = True
    SUPPORTS_SCENE_QUERY_FNS = True
    SUPPORTS_INTERACTION_FNS = True
    SUPPORTS_BENCHMARK_FNS = True

    def setup_method(self) -> None:
        """Resolve the process-wide OvPhysX simulation identifier."""
        self.simulation_id = _ps.find_ovphysx_sim_id()
        if self.simulation_id is None:
            pytest.skip("OvPhysX backend not registered")

    def teardown_method(self) -> None:
        """Leave the process-wide OvPhysX registration intact."""
