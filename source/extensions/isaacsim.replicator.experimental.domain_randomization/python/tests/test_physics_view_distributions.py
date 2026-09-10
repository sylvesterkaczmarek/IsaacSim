# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Verifies physics-view randomization graph authoring with Replicator distributions."""

import carb.settings
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.replicator.experimental.domain_randomization as dr
import omni.kit.test
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.simulation_manager import SimulationManager


class TestPhysicsViewDistributions(omni.kit.test.AsyncTestCase):
    """Test suite for physics-view randomization distribution wiring."""

    async def setUp(self) -> None:
        """Create a clean stage and save render and physics-device settings changed by the test."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()
        self.original_dlss_exec_mode = carb.settings.get_settings().get("rtx/post/dlss/execMode")
        self.original_physics_sim_device = SimulationManager.get_device()
        SimulationManager.setup_simulation()
        await app_utils.update_app_async()
        dr.physics_view.register_simulation_context()

    async def tearDown(self) -> None:
        """Close the stage, wait for pending loads, and restore render and physics-device settings."""
        dr.physics_view.cleanup()
        stage_utils.close_stage()
        await app_utils.update_app_async()
        # In some cases the test will end before the asset is loaded, in this case wait for assets to load
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await app_utils.update_app_async()
        carb.settings.get_settings().set("rtx/post/dlss/execMode", self.original_dlss_exec_mode)
        # Make sure to reset the physics sim device to the original state for the following tests
        SimulationManager.set_physics_sim_device(self.original_physics_sim_device)

    async def test_randomize_simulation_context_accepts_sequence_distribution(self) -> None:
        """Author a simulation-context writer graph with a sequence distribution."""
        with dr.trigger.on_rl_frame(num_envs=1):
            with dr.gate.on_interval(interval=20):
                dr.physics_view.randomize_simulation_context(
                    operation="scaling",
                    gravity=rep.distribution.sequence([(1.0, 1.0, 1.0), (1.0, 1.0, 2.0)], ordered=True),
                )
        dr.physics_view.step_randomization()
        await app_utils.update_app_async()
