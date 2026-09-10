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

"""Deterministic tests for scenario-owned planning state."""

from unittest.mock import MagicMock

import omni.kit.test
from isaacsim.robot_motion.examples.manipulation import ManipulationScenario


class TestManipulationScenario(omni.kit.test.AsyncTestCase):
    async def test_robot_prim_path_must_be_an_absolute_prim_path(self) -> None:
        for path in ("robot", "/World/robot.visibility"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                ManipulationScenario(robot_prim_path=path)

    async def test_compound_obstacle_updates_are_owned_and_atomic(self) -> None:
        scenario = ManipulationScenario()
        scenario._tracked_collision_paths = (
            "/World/Cube/mesh_a",
            "/World/Cube/mesh_b",
            "/World/Other/mesh",
        )
        interface = MagicMock()
        scenario._world_binding = MagicMock()
        scenario._world_binding.get_world_interface.return_value = interface

        scenario.set_planning_obstacles_enabled(["/World/Cube"], False)
        call = interface.update_obstacle_enables.call_args
        self.assertEqual(call.kwargs["prim_paths"], ["/World/Cube/mesh_a", "/World/Cube/mesh_b"])
        self.assertEqual(call.kwargs["enabled_array"].numpy().tolist(), [False, False])

        interface.reset_mock()
        with self.assertRaises(ValueError):
            scenario.set_planning_obstacles_enabled(["/World/Cube", "/World/Missing"], True)
        interface.update_obstacle_enables.assert_not_called()
