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

"""Verifies stage close clears domain-randomization registries and context."""

import carb.settings
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.replicator.experimental.domain_randomization as dr
import omni.ext
import omni.kit.test
import omni.usd
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.replicator.experimental.domain_randomization.scripts import context as dr_context


class TestCleanupOnStageClose(omni.kit.test.AsyncTestCase):
    """Test suite for stage-close cleanup of physics-view registries and context."""

    async def setUp(self) -> None:
        """Create a clean stage and save render and physics-device settings changed by the test."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()
        self.original_dlss_exec_mode = carb.settings.get_settings().get("rtx/post/dlss/execMode")
        self.original_physics_sim_device = SimulationManager.get_device()

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

    async def test_extension_is_exported(self) -> None:
        """Export ``Extension`` on the package so Kit can construct the IExt."""
        self.assertTrue(hasattr(dr, "Extension"))
        self.assertTrue(issubclass(dr.Extension, omni.ext.IExt))

    async def test_stage_close_clears_view_registries_and_context(self) -> None:
        """Clear view registries and the session context when the stage closes."""
        dr.physics_view._articulation_views["franka_view"] = object()
        dr.physics_view._articulation_views_reset_values["franka_view"] = {}
        dr.physics_view._articulation_views_initial_values["franka_view"] = {}
        dr.physics_view._rigid_prim_views["object_view"] = object()
        dr.physics_view._rigid_prim_views_reset_values["object_view"] = {}
        dr.physics_view._rigid_prim_views_initial_values["object_view"] = {}
        dr_context._context = object()

        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()

        self.assertEqual(dr.physics_view._articulation_views, {})
        self.assertEqual(dr.physics_view._articulation_views_reset_values, {})
        self.assertEqual(dr.physics_view._articulation_views_initial_values, {})
        self.assertEqual(dr.physics_view._rigid_prim_views, {})
        self.assertEqual(dr.physics_view._rigid_prim_views_reset_values, {})
        self.assertEqual(dr.physics_view._rigid_prim_views_initial_values, {})
        self.assertIsNone(dr_context._context)

    async def test_context_apis_require_initialized_session(self) -> None:
        """Raise a descriptive error when context APIs run without an active session."""
        dr_context._context = None
        with self.assertRaisesRegex(RuntimeError, "Domain randomization context is not initialized"):
            dr_context.get_reset_inds()
        with self.assertRaisesRegex(RuntimeError, "Domain randomization context is not initialized"):
            dr_context.trigger_randomization([0])
