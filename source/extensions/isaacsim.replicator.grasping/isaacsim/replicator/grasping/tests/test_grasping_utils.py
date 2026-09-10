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

"""Test grasping utility helpers."""

from unittest.mock import Mock, patch

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
import omni.usd
from isaacsim.replicator.grasping import grasping_utils
from pxr import PhysxSchema, Usd, UsdPhysics


class TestGraspingUtils(omni.kit.test.AsyncTestCase):
    """Test grasping utility functions."""

    async def setUp(self) -> None:
        """Create a clean stage."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        """Close the stage and wait for pending asset loads."""
        stage_utils.close_stage()
        await app_utils.update_app_async()
        # In some cases the test will end before the asset is loaded, in this case wait for assets to load
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await app_utils.update_app_async()

    def _create_physics_scene(self) -> UsdPhysics.Scene:
        return UsdPhysics.Scene.Define(stage_utils.get_current_stage(), "/World/PhysicsScene")

    def _get_update_type_attr(self, physics_scene: UsdPhysics.Scene) -> Usd.Attribute:
        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
        return physx_scene_api.GetUpdateTypeAttr()

    def _create_update_type_attr(self, physics_scene: UsdPhysics.Scene) -> Usd.Attribute:
        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
        return physx_scene_api.CreateUpdateTypeAttr()

    def _has_authored_update_type(self, physics_scene: UsdPhysics.Scene) -> bool:
        update_type_attr = self._get_update_type_attr(physics_scene)
        return bool(update_type_attr) and update_type_attr.HasAuthoredValueOpinion()

    async def test_simulate_physics_async_restores_authored_update_type(self) -> None:
        """Test that scene simulation restores an authored PhysX update type."""
        physics_scene = self._create_physics_scene()
        update_type_attr = self._create_update_type_attr(physics_scene)
        update_type_attr.Set(PhysxSchema.Tokens.Synchronous)

        physx_sim_interface = Mock()
        with patch.object(
            grasping_utils.omni.physx, "get_physx_simulation_interface", return_value=physx_sim_interface
        ):
            await grasping_utils.simulate_physics_async(
                num_frames=1, step_dt=1.0 / 60.0, physics_scene=physics_scene, render=False
            )

        self.assertEqual(update_type_attr.Get(), PhysxSchema.Tokens.Synchronous)
        self.assertTrue(update_type_attr.HasAuthoredValueOpinion())

    async def test_simulate_physics_async_clears_unauthored_update_type(self) -> None:
        """Test that scene simulation clears PhysX update type when it was originally unauthored."""
        physics_scene = self._create_physics_scene()
        self.assertFalse(self._has_authored_update_type(physics_scene))

        physx_sim_interface = Mock()
        with patch.object(
            grasping_utils.omni.physx, "get_physx_simulation_interface", return_value=physx_sim_interface
        ):
            await grasping_utils.simulate_physics_async(
                num_frames=1, step_dt=1.0 / 60.0, physics_scene=physics_scene, render=False
            )

        self.assertFalse(self._has_authored_update_type(physics_scene))

    async def test_simulate_physics_with_forces_async_restores_authored_update_type(self) -> None:
        """Test that force simulation restores an authored PhysX update type."""
        physics_scene = self._create_physics_scene()
        update_type_attr = self._create_update_type_attr(physics_scene)
        update_type_attr.Set(PhysxSchema.Tokens.Synchronous)

        physx_sim_interface = Mock()
        with patch.object(
            grasping_utils.omni.physx, "get_physx_simulation_interface", return_value=physx_sim_interface
        ):
            await grasping_utils.simulate_physics_with_forces_async(
                stage_id=0,
                body_ids=[],
                forces=[],
                positions=[],
                sim_steps=1,
                physx_dt=1.0 / 60.0,
                physics_scene=physics_scene,
                render=False,
            )

        self.assertEqual(update_type_attr.Get(), PhysxSchema.Tokens.Synchronous)
        self.assertTrue(update_type_attr.HasAuthoredValueOpinion())

    async def test_simulate_physics_with_forces_async_restores_update_type_on_error(self) -> None:
        """Test that force simulation restores PhysX update type if stepping fails."""
        physics_scene = self._create_physics_scene()
        update_type_attr = self._create_update_type_attr(physics_scene)
        update_type_attr.Set(PhysxSchema.Tokens.Synchronous)

        physx_sim_interface = Mock()
        physx_sim_interface.simulate_scene.side_effect = RuntimeError("Simulation failed.")
        with patch.object(
            grasping_utils.omni.physx, "get_physx_simulation_interface", return_value=physx_sim_interface
        ):
            with self.assertRaises(RuntimeError):
                await grasping_utils.simulate_physics_with_forces_async(
                    stage_id=0,
                    body_ids=[],
                    forces=[],
                    positions=[],
                    sim_steps=1,
                    physx_dt=1.0 / 60.0,
                    physics_scene=physics_scene,
                    render=False,
                )

        self.assertEqual(update_type_attr.Get(), PhysxSchema.Tokens.Synchronous)
        self.assertTrue(update_type_attr.HasAuthoredValueOpinion())
