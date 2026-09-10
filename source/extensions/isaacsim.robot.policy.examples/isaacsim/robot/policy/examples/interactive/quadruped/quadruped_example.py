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

"""Interactive quadruped robot control example using reinforcement learning policies and keyboard input."""

from isaacsim.robot.policy.examples import get_spot_spec
from isaacsim.robot.policy.examples.interactive.example_base import (
    LocomotionPolicySample,
    make_velocity_keyboard_mapping,
)


class QuadrupedExample(LocomotionPolicySample):
    """Boston Dynamics Spot locomotion example with GPU-accelerated physics and keyboard control.

    The policy is deployed through the generic ``RobotPolicyRunner`` with the bundled Spot spec.
    Arrow keys or NUMPAD 8/2/4/6 command planar movement; N/M or NUMPAD 7/9 command yaw.
    """

    _spec_getter = staticmethod(get_spot_spec)
    _prim_path = "/World/Spot"
    _spawn_position = (0.0, 0.0, 0.8)
    _physics_dt = 1.0 / 500.0  # 500 Hz physics
    _rendering_dt = 10.0 / 500.0  # 50 Hz rendering (10 physics steps per render)
    _input_keyboard_mapping = make_velocity_keyboard_mapping(2.0)
