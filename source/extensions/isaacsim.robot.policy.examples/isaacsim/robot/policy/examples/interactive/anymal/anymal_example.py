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

"""ANYmal robot locomotion example with keyboard control."""

from isaacsim.robot.policy.examples import get_anymal_spec
from isaacsim.robot.policy.examples.interactive.example_base import (
    LocomotionPolicySample,
    make_velocity_keyboard_mapping,
)


class AnymalExample(LocomotionPolicySample):
    """ANYmal robot locomotion example with keyboard control.

    The physics engine is determined by the currently active engine in SimulationManager
    (switchable via the viewport's Physics Engine menu before loading). The policy is
    deployed through the generic ``RobotPolicyRunner`` with the bundled ANYmal spec.
    """

    _spec_getter = staticmethod(get_anymal_spec)
    _prim_path = "/World/Anymal"
    _spawn_position = (0.0, 0.0, 0.7)
    _physics_dt = 1.0 / 200.0  # 200 Hz physics (matches training)
    _rendering_dt = 8.0 / 200.0  # 25 Hz rendering (8 physics steps per render)
    _input_keyboard_mapping = make_velocity_keyboard_mapping(1.0)
