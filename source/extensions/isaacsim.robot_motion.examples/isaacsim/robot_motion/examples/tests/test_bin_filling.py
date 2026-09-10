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

"""Runtime coverage for the code-authored bin-filling example."""

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
from isaacsim.robot_motion.examples.manipulation.interactive.bin_filling import BinFilling
from pxr import Gf


class TestBinFilling(omni.kit.test.AsyncTestCase):
    async def test_scene_runs_and_resets(self) -> None:
        sample = BinFilling()
        try:
            await sample.load_world_async()

            stage = stage_utils.get_current_stage(backend="usd")
            for prim_path in (
                "/World/Scene/ur10/ee_link/SurfaceGripper",
                "/World/Scene/bin/Collision/Cube_03",
                "/World/Scene/pipe/pipe_funnel",
                "/World/Scene/gridroom_curved",
            ):
                self.assertTrue(stage.GetPrimAtPath(prim_path).IsValid(), prim_path)

            bin_prim = stage.GetPrimAtPath("/World/Scene/bin")
            self.assertAlmostEqual(bin_prim.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get(), 0.0005)
            self.assertAlmostEqual(bin_prim.GetAttribute("physxRigidBody:sleepThreshold").Get(), 0.002)
            self.assertAlmostEqual(bin_prim.GetAttribute("physxRigidBody:stabilizationThreshold").Get(), 1.0e-8)
            for child_name in ("Visuals", "Collision"):
                child = stage.GetPrimAtPath(f"/World/Scene/bin/{child_name}")
                self.assertEqual(child.GetAttribute("xformOp:scale").Get(), Gf.Vec3d(0.992, 0.992, 1.073))

            funnel = stage.GetPrimAtPath("/World/Scene/pipe/pipe_funnel")
            self.assertEqual(funnel.GetAttribute("xformOp:orient").Get(), Gf.Quatd(0.5, 0.5, 0.5, 0.5))

            funnel_component = stage.GetPrimAtPath("/World/Scene/pipe/pipe_funnel/Cylinder_004_Cylinder_008")
            self.assertEqual(funnel_component.GetAttribute("xformOp:orient").Get(), Gf.Quatd(0.0, 1.0, 0.0, 0.0))
            self.assertEqual(funnel_component.GetAttribute("xformOp:scale").Get(), Gf.Vec3d(0.01))

            initial_position = sample._bin_prim.get_world_poses()[0].numpy()[0]
            await sample.on_fill_bin_event_async()
            await app_utils.update_app_async(steps=1500, callback=lambda _step, _steps: sample._event < 5)
            self.assertGreaterEqual(sample._event, 5)
            lifted_position = sample._bin_prim.get_world_poses()[0].numpy()[0]
            self.assertGreater(lifted_position[2], initial_position[2] + 0.05)

            await sample.reset_async()
            self.assertEqual(sample._event, 0)
        finally:
            await sample.clear_async()
