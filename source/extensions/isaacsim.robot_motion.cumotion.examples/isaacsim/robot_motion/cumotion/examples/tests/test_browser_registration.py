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

"""Tests for cuMotion Robotics Examples browser registration."""

import omni.kit.test
from isaacsim.examples.browser import get_instance as get_browser_instance


class TestBrowserRegistration(omni.kit.test.AsyncTestCase):
    """Verify that all cuMotion examples are discoverable in the expected folder."""

    async def test_cumotion_examples_are_registered(self) -> None:
        """Check the exact example names and that each entry supplies a UI hook."""
        entries = get_browser_instance()._browser_model._examples["Motion Generation/cuMotion"]
        self.assertEqual(
            {entry.example.name for entry in entries},
            {
                "Graph Planner",
                "RMPflow",
                "Trajectory Generator",
                "Trajectory Optimizer",
                "World Interface",
            },
        )
        self.assertEqual(len(entries), 5)
        self.assertTrue(all(callable(entry.ui_hook) for entry in entries))
