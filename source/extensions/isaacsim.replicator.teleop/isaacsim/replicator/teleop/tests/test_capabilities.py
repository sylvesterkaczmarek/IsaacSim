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

"""Tests for optional teleop backend discovery."""

from unittest.mock import patch

import omni.kit.test
from isaacsim.replicator.teleop import get_teleop_capabilities


class TestTeleopCapabilities(omni.kit.test.AsyncTestCase):
    """Verify native backends remain optional while core features stay available."""

    async def test_missing_native_backends_preserve_debug_and_scripted_features(self) -> None:
        """Report partial support without importing optional modules."""

        def module_available(name: str) -> bool:
            return False

        with (
            patch("isaacsim.replicator.teleop.capabilities._module_available", side_effect=module_available),
            patch("isaacsim.replicator.teleop.capabilities.platform.system", return_value="Windows"),
        ):
            capabilities = get_teleop_capabilities()

        self.assertTrue(capabilities.debug_input)
        self.assertTrue(capabilities.scripted_motion)
        self.assertFalse(capabilities.live_input)
        self.assertFalse(capabilities.mcap_replay)
        self.assertFalse(capabilities.pink_ik)
        self.assertIn("Debug Mode", capabilities.native_input_unavailable_reason)
        self.assertIn("Windows", capabilities.pink_ik_unavailable_reason)

    async def test_installed_optional_backends_are_reported(self) -> None:
        """Expose live, replay, and PINK support when both packages resolve."""
        with patch("isaacsim.replicator.teleop.capabilities._module_available", return_value=True):
            capabilities = get_teleop_capabilities()

        self.assertTrue(capabilities.live_input)
        self.assertTrue(capabilities.mcap_replay)
        self.assertTrue(capabilities.pink_ik)
        self.assertEqual(capabilities.native_input_unavailable_reason, "")
        self.assertEqual(capabilities.pink_ik_unavailable_reason, "")
