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

"""Extension that demonstrates a Franka Panda robot performing a drawer opening task using a trained policy."""

import os

from isaacsim.robot.policy.examples.interactive.example_base import PolicyExampleExtension
from isaacsim.robot.policy.examples.interactive.franka.franka_example import FrankaExample


class FrankaExampleExtension(PolicyExampleExtension):
    """Register the Franka Panda drawer-opening policy example in the examples browser."""

    example_name = "Franka"
    title = "Manipulator: Franka"
    sample_class = FrankaExample
    file_path = os.path.abspath(__file__)
    overview = (
        "This Example shows a Franka Panda open drawer policy trained in Isaac Lab. "
        "The Franka will attempt to open the drawer in front of it and hold it open. "
        "The scene will reset every 10s (sim time) and the Franka will try again."
    )
