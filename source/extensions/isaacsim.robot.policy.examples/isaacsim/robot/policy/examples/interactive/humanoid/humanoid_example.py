# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Interactive humanoid robot simulation example using H1 robot with GPU-accelerated physics and keyboard control."""

from isaacsim.robot.policy.examples import get_h1_spec
from isaacsim.robot.policy.examples.interactive.example_base import LocomotionPolicySample


class HumanoidExample(LocomotionPolicySample):
    """Unitree H1 humanoid locomotion example with GPU-accelerated physics and keyboard control.

    The policy is deployed through the generic ``RobotPolicyRunner`` with the bundled H1 spec.
    NUMPAD_8/UP moves forward; NUMPAD_4/LEFT and NUMPAD_6/RIGHT turn left and right.
    """

    _spec_getter = staticmethod(get_h1_spec)
    _prim_path = "/World/H1"
    _spawn_position = (0.0, 0.0, 1.05)
    _physics_dt = 1.0 / 200.0  # 200 Hz physics
    _rendering_dt = 8.0 / 200.0  # 25 Hz rendering (8 physics steps per render)
    _input_keyboard_mapping = {
        # forward command
        "NUMPAD_8": [0.75, 0.0, 0.0],
        "UP": [0.75, 0.0, 0.0],
        # yaw command (positive)
        "NUMPAD_4": [0.0, 0.0, 0.75],
        "LEFT": [0.0, 0.0, 0.75],
        # yaw command (negative)
        "NUMPAD_6": [0.0, 0.0, -0.75],
        "RIGHT": [0.0, 0.0, -0.75],
    }
