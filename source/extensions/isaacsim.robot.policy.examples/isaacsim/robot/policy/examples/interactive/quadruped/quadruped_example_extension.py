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

"""Extension demonstrating quadruped robot control using a Boston Dynamics Spot with a trained policy."""

import os

from isaacsim.robot.policy.examples.interactive.example_base import PolicyExampleExtension
from isaacsim.robot.policy.examples.interactive.quadruped.quadruped_example import QuadrupedExample


class QuadrupedExampleExtension(PolicyExampleExtension):
    """Register the Boston Dynamics Spot quadruped locomotion example in the examples browser."""

    example_name = "Quadruped"
    title = "Quadruped: Boston Dynamics Spot"
    sample_class = QuadrupedExample
    file_path = os.path.abspath(__file__)
    overview = (
        "This Example shows a Boston Dynamics Spot running a flat terrain policy trained in Isaac Lab. "
        "Use the Physics Engine menu in the viewport to switch between PhysX and Newton before loading. "
        "\n\n\tKeyboard Input:"
        "\n\t\tup arrow / numpad 8: Move Forward"
        "\n\t\tdown arrow/ numpad 2: Move Reverse"
        "\n\t\tleft arrow/ numpad 4: Move Left"
        "\n\t\tright arrow / numpad 6: Move Right"
        "\n\t\tN / numpad 7: Spin Counterclockwise"
        "\n\t\tM / numpad 9: Spin Clockwise"
        "\n\nPress the 'Open in IDE' button to view the source code."
    )
