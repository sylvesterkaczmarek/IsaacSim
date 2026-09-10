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

"""Go2 robot locomotion example with keyboard control."""

from isaacsim.robot.policy.examples import get_go2_spec
from isaacsim.robot.policy.examples.interactive.example_base import (
    LocomotionPolicySample,
    make_velocity_keyboard_mapping,
)


class Go2Example(LocomotionPolicySample):
    """Go2 robot locomotion example with keyboard control.

    The physics engine is determined by the currently active engine in SimulationManager
    (switchable via the viewport's Physics Engine menu before loading). The policy is
    deployed through the generic ``RobotPolicyRunner`` with the bundled Go2 spec.
    """

    _spec_getter = staticmethod(get_go2_spec)
    _prim_path = "/World/Go2"
    _spawn_position = (0.0, 0.0, 0.5)
    _physics_dt = 0.005  # 200 Hz physics (matches training)
    _rendering_dt = 0.02  # 50 Hz rendering (4 physics steps per render)
    _input_keyboard_mapping = make_velocity_keyboard_mapping(1.5)
