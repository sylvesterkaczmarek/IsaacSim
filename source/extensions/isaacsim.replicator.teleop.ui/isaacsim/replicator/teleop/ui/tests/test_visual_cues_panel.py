# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for Visual Cues panel validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import omni.kit.test
from isaacsim.replicator.teleop.ui.visual_cues_panel import VisualCuesPanel


class TestVisualCuesPanel(omni.kit.test.AsyncTestCase):
    """Verify invalid manual overrides cannot activate stale cue sources."""

    async def test_show_rejects_invalid_override(self) -> None:
        """Keep a cue hidden when its current manual prim path is invalid."""
        manager = mock.Mock()
        teleop_manager = mock.Mock()
        teleop_manager.get_input_world_position = mock.Mock()
        panel = VisualCuesPanel(manager, teleop_manager, {})
        path_model = mock.Mock()
        path_model.get_value_as_string.return_value = "/World/Missing"
        panel._widgets["left"] = {  # noqa: SLF001
            "path": SimpleNamespace(model=path_model),
            "status": None,
        }

        with mock.patch.object(panel, "_validate_override_path", return_value=False):
            panel._on_show("left")  # noqa: SLF001

        manager.set_override_prim_path.assert_not_called()
        manager.show_side.assert_not_called()
        self.assertFalse(panel._desired_enabled["left"])  # noqa: SLF001
