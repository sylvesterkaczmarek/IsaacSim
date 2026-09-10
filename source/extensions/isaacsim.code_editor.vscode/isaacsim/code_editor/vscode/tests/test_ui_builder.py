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

"""Tests for live Python server endpoint reporting."""

from unittest.mock import patch

import omni.kit.test
from isaacsim.code_editor.python_server import BoundEndpoint, ServerState, ServerStatus
from isaacsim.code_editor.vscode.ui_builder import UIBuilder


def _status(state: ServerState, error: str | None = None) -> ServerStatus:
    endpoints = (BoundEndpoint("127.0.0.1", 9001),) if state == ServerState.RUNNING else ()
    return ServerStatus(
        state=state,
        configured_host="127.0.0.1",
        configured_port=9001,
        bound_endpoints=endpoints,
        active_connections=0,
        authentication_required=False,
        last_error=error,
    )


class TestUIBuilderServerStatus(omni.kit.test.AsyncTestCase):
    """The launcher notification must describe live rather than cached state."""

    async def test_running_uses_bound_endpoint(self) -> None:
        builder = UIBuilder.__new__(UIBuilder)
        with patch("isaacsim.code_editor.python_server.get_server_status", return_value=_status(ServerState.RUNNING)):
            message, warning = builder._get_server_notification()
        self.assertEqual(message, "Serving at 127.0.0.1:9001")
        self.assertFalse(warning)

    async def test_error_never_claims_serving(self) -> None:
        builder = UIBuilder.__new__(UIBuilder)
        with patch(
            "isaacsim.code_editor.python_server.get_server_status",
            return_value=_status(ServerState.ERROR, "address in use"),
        ):
            message, warning = builder._get_server_notification()
        self.assertIn("error", message)
        self.assertIn("address in use", message)
        self.assertNotIn("Serving at", message)
        self.assertTrue(warning)

    async def test_unavailable_status_has_no_cached_endpoint_claim(self) -> None:
        builder = UIBuilder.__new__(UIBuilder)
        with patch("isaacsim.code_editor.python_server.get_server_status", return_value=None):
            message, warning = builder._get_server_notification()
        self.assertEqual(message, "Python server status unavailable")
        self.assertTrue(warning)
