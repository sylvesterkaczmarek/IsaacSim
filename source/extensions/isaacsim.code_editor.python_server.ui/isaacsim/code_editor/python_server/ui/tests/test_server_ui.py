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

"""UI smoke tests for the Python server indicator and management modal."""

import asyncio
from dataclasses import replace

import omni.kit.app
import omni.kit.test
import omni.kit.ui_test as ui_test
import omni.ui as ui
from isaacsim.code_editor.python_server import ServerState
from isaacsim.code_editor.python_server.ui.indicator import PythonServerDelegate


class TestPythonServerUi(omni.kit.test.AsyncTestCase):
    """Build the delegate in an isolated window and verify stable identifiers."""

    async def setUp(self) -> None:
        """Create an isolated indicator host window."""
        self.window = ui.Window("Python Server UI Test", width=500, height=100)
        self.delegate = PythonServerDelegate()
        with self.window.frame:
            self.delegate.build_item(None)
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

    async def tearDown(self) -> None:
        """Destroy the delegate and host window."""
        self.delegate.destroy()
        self.window.destroy()

    async def test_indicator_opens_management_modal(self) -> None:
        """Clicking the indicator opens the management modal."""
        indicator = ui_test.find(
            "Python Server UI Test//Frame/**/InvisibleButton[*].identifier=='python_server_indicator'"
        )
        self.assertIsNotNone(indicator)
        await indicator.click()
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()
        self.assertIsNotNone(ui_test.find("Python Server"))
        host = ui_test.find("Python Server//Frame/**/StringField[*].identifier=='python_server_host'")
        port = ui_test.find("Python Server//Frame/**/IntField[*].identifier=='python_server_port'")
        toggle = ui_test.find("Python Server//Frame/**/Button[*].identifier=='python_server_toggle'")
        close = ui_test.find("Python Server//Frame/**/Button[*].identifier=='python_server_close'")
        self.assertIsNotNone(host)
        self.assertIsNotNone(port)
        self.assertIsNotNone(toggle)
        self.assertIsNotNone(close)

    async def test_indicator_is_registered_in_menu_bar(self) -> None:
        """The UI extension registers its right-aligned menu delegate."""
        menu = ui_test.get_menubar().find_menu("Python Server Status Widget")
        self.assertIsNotNone(menu)

    async def test_only_one_ui_action_runs_at_a_time(self) -> None:
        """The delegate serializes lifecycle actions."""
        started = 0
        release = asyncio.Event()

        async def action() -> None:
            nonlocal started
            started += 1
            await release.wait()

        self.delegate._schedule(action)
        task = self.delegate._task
        self.delegate._schedule(action)
        await omni.kit.app.get_app().next_update_async()
        self.assertEqual(started, 1)
        release.set()
        await task
        await omni.kit.app.get_app().next_update_async()
        self.assertIsNone(self.delegate._task)

    async def test_error_indicator_opens_and_shows_error_details(self) -> None:
        """An error snapshot is visible in the popup and tooltip."""
        expected_error = "address in use"
        status = replace(
            self.delegate._status,
            state=ServerState.ERROR,
            bound_endpoints=(),
            last_error=expected_error,
        )
        self.delegate._on_status(status)

        indicator = ui_test.find(
            "Python Server UI Test//Frame/**/InvisibleButton[*].identifier=='python_server_indicator'"
        )
        await indicator.click()
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

        self.assertTrue(self.delegate._popup.visible)
        error = ui_test.find("Python Server//Frame/**/Label[*].identifier=='python_server_error'")
        self.assertIsNotNone(error)
        self.assertEqual(error.widget.text, expected_error)
        self.assertIn(expected_error, self.delegate._button.tooltip)

    async def test_shared_port_warning_is_visible(self) -> None:
        """A shared Linux listener is clearly reported by the indicator."""
        if self.delegate._port_monitor_task is not None:
            self.delegate._port_monitor_task.cancel()
        self.delegate._is_port_shared = lambda _port: True
        self.delegate._port_monitor_task = asyncio.create_task(self.delegate._monitor_port())
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
            if self.delegate._port_shared:
                break
        self.assertIn("Port 8227 is shared with another process.", self.delegate._button.tooltip)
