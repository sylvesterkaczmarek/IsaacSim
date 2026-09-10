# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test API calls from SimulationView."""

import numpy as np

from .test_helpers import NewtonTensorTestBase, run_on_device_configs


@run_on_device_configs()
class TestSimulationView(NewtonTensorTestBase):
    """Test Simulation View."""

    async def test_simulation_view_gravity(self) -> None:
        """Set gravity using tensor view, then checking it again with tensor view."""
        self.setup_ball_grid(num_envs=2)
        sim = await self.create_sim()
        sim.set_gravity([0.1, 0.2, 0.3])
        gravity = sim.get_gravity()
        self.assertTrue(np.allclose(np.array([0.1, 0.2, 0.3]), np.array(gravity)))
