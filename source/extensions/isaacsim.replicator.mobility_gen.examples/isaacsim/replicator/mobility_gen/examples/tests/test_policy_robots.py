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

"""Tests for the MobilityGen H1 and Spot RobotPolicyRunner boundary."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import omni.kit.test
from isaacsim.replicator.mobility_gen.examples import robots as robots_module


class _PolicyRobot(robots_module.PolicyMobilityGenRobot):
    z_offset = 0.7
    runner = None

    @classmethod
    def build_policy(cls, prim_path: str) -> object:
        return cls.runner

    @classmethod
    def build_front_camera(cls, prim_path: str) -> None:
        return None


class _PolicyMultiRobot(robots_module.PolicyMultiSensorRobot):
    z_offset = 0.7
    runner = None

    @classmethod
    def build_policy(cls, prim_path: str) -> object:
        return cls.runner

    @classmethod
    def build_sensor_rig(cls, prim_path: str) -> None:
        return None


class TestPolicyMobilityGenRobot(omni.kit.test.AsyncTestCase):
    """TestPolicyMobilityGenRobot."""

    def test_policy_build_uses_the_runner_asset_and_articulation(self) -> None:
        """Verify policy build uses the runner asset and articulation."""
        for robot_type in (_PolicyRobot, _PolicyMultiRobot):
            with self.subTest(robot_type=robot_type.__name__):
                articulation = object()
                runner = MagicMock()
                runner.spawn.return_value = articulation
                robot_type.runner = runner
                stage = MagicMock()
                with (
                    patch.object(robots_module, "add_reference_to_stage") as add_reference,
                    patch.object(robots_module, "get_current_stage", return_value=stage),
                ):
                    robot = robot_type.build("/World/robot")

                add_reference.assert_not_called()
                runner.spawn.assert_called_once_with()
                stage.Load.assert_called_once_with("/World/robot")
                self.assertIs(robot.articulation, articulation)

    def test_policy_action_initializes_once_and_steps_each_tick(self) -> None:
        """Verify policy action initializes once and steps each tick."""
        for robot_type in (_PolicyRobot, _PolicyMultiRobot):
            with self.subTest(robot_type=robot_type.__name__):
                runner = MagicMock()
                robot = robot_type("/World/robot", MagicMock(), runner)
                robot.action.set_value(np.array([1.25, -0.5]))

                robot.write_action(0.005)
                robot.write_action(0.005)

                runner.initialize.assert_called_once_with()
                runner.step.assert_has_calls(
                    [
                        call(0.005, [1.25, 0.0, -0.5]),
                        call(0.005, [1.25, 0.0, -0.5]),
                    ]
                )
