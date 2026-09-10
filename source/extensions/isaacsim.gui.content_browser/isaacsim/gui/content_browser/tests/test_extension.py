# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The Kit extension system tests for Python has additional wrapping.

to make test auto-discoverable add support for async/await tests.
The easiest way to set up the test class is to have it derive from
the omni.kit.test.AsyncTestCase class that implements them.

Visit the next link for more details:
  https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/testing_exts_python.html
"""

import asyncio
import weakref
from pathlib import Path
from unittest.mock import Mock, call, patch

import carb.settings
import omni.kit.test
from isaacsim.gui.content_browser.impl.extension import Extension
from isaacsim.gui.content_browser.impl.isaac_collection import SETTING_ASSET_ROOT, IsaacCollection
from omni.kit.window.file_importer import get_file_importer


class TestExtension(omni.kit.test.AsyncTestCase):
    """Test suite for the content browser extension."""

    async def setUp(self) -> None:
        """Method called to prepare the test fixture."""
        super().setUp()
        # ---------------
        # Do custom setUp
        # ---------------

    async def tearDown(self) -> None:
        """Method called immediately after the test method has been called."""
        file_importer = get_file_importer()
        if file_importer and file_importer.get_dialog():
            file_importer.hide_window()
            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()
        super().tearDown()

    # --------------------------------------------------------------------

    async def test_extension(self) -> None:
        """Test that the extension loads successfully."""
        # Kit extension system test for Python is based on the unittest module.
        # Visit https://docs.python.org/3/library/unittest.html to see the
        # available assert methods to check for and report failures.
        print("Test case: test_extension")
        self.assertTrue(True)

    @patch("isaacsim.gui.content_browser.impl.extension.IsaacCollection")
    @patch("isaacsim.gui.content_browser.impl.extension.get_file_importer")
    async def test_file_importer_registers_isaac_collection_once_per_dialog(
        self, get_file_importer_mock: Mock, isaac_collection_mock: Mock
    ) -> None:
        """Test that a File Importer dialog receives one Isaac Sim collection."""
        extension = Extension()
        extension._file_picker_collection = None
        extension._file_picker_dialog_ref = None

        dialog = Mock()
        dialog._widget.api.register_collection_item.return_value = True
        get_file_importer_mock.return_value.get_dialog.return_value = dialog

        collection = Mock()
        isaac_collection_mock.return_value = collection

        extension._on_file_picker_ui_ready(Mock())
        extension._on_file_picker_ui_ready(Mock())

        dialog._widget.api.register_collection_item.assert_called_once_with(collection)
        self.assertIs(extension._file_picker_collection, collection)
        self.assertIs(extension._file_picker_dialog_ref(), dialog)

    @patch("isaacsim.gui.content_browser.impl.extension.IsaacCollection")
    @patch("isaacsim.gui.content_browser.impl.extension.get_file_importer")
    async def test_file_importer_replaces_collection_for_each_new_dialog(
        self, get_file_importer_mock: Mock, isaac_collection_mock: Mock
    ) -> None:
        """Test that recreated File Importer dialogs replace the tracked collection."""
        extension = Extension()
        extension._file_picker_collection = None
        extension._file_picker_dialog_ref = None

        first_dialog = Mock()
        first_dialog._widget.api.register_collection_item.return_value = True
        second_dialog = Mock()
        second_dialog._widget.api.register_collection_item.return_value = True
        get_file_importer_mock.return_value.get_dialog.side_effect = [first_dialog, second_dialog]

        first_collection = Mock()
        second_collection = Mock()
        isaac_collection_mock.side_effect = [first_collection, second_collection]

        extension._on_file_picker_ui_ready(Mock())
        extension._on_file_picker_ui_ready(Mock())

        first_dialog._widget.api.register_collection_item.assert_called_once_with(first_collection)
        first_dialog._widget.api.deregister_collection_item.assert_called_once_with(first_collection)
        second_dialog._widget.api.register_collection_item.assert_called_once_with(second_collection)
        self.assertIs(extension._file_picker_collection, second_collection)
        self.assertIs(extension._file_picker_dialog_ref(), second_dialog)

    @patch("isaacsim.gui.content_browser.impl.extension.IsaacCollection")
    @patch("isaacsim.gui.content_browser.impl.extension.get_file_importer")
    async def test_file_importer_retries_failed_collection_registration(
        self, get_file_importer_mock: Mock, isaac_collection_mock: Mock
    ) -> None:
        """Test that a rejected registration does not mark the dialog as complete."""
        extension = Extension()
        extension._file_picker_collection = None
        extension._file_picker_dialog_ref = None

        dialog = Mock()
        dialog._widget.api.register_collection_item.side_effect = [False, True]
        get_file_importer_mock.return_value.get_dialog.return_value = dialog

        first_collection = Mock()
        second_collection = Mock()
        isaac_collection_mock.side_effect = [first_collection, second_collection]

        extension._on_file_picker_ui_ready(Mock())
        self.assertIsNone(extension._file_picker_collection)
        self.assertIsNone(extension._file_picker_dialog_ref)

        extension._on_file_picker_ui_ready(Mock())

        self.assertEqual(
            dialog._widget.api.register_collection_item.call_args_list,
            [call(first_collection), call(second_collection)],
        )
        self.assertIs(extension._file_picker_collection, second_collection)
        self.assertIs(extension._file_picker_dialog_ref(), dialog)

    async def test_file_importer_deregisters_collection_on_shutdown(self) -> None:
        """Test that extension shutdown removes its collection from an open dialog."""
        extension = Extension()
        dialog = Mock()
        collection = Mock()
        extension._content_browser_ref = Mock(return_value=None)
        extension._file_picker_collection = collection
        extension._file_picker_dialog_ref = weakref.ref(dialog)
        extension._file_picker_ui_ready_sub = Mock()

        extension.on_shutdown()

        dialog._widget.api.deregister_collection_item.assert_called_once_with(collection)
        self.assertIsNone(extension._file_picker_collection)
        self.assertIsNone(extension._file_picker_dialog_ref)
        self.assertIsNone(extension._file_picker_ui_ready_sub)

    async def test_isaac_collection_uses_configured_asset_root(self) -> None:
        """Test that collection folders resolve against the configured asset root."""
        settings = carb.settings.get_settings()
        original_asset_root = settings.get_as_string(SETTING_ASSET_ROOT)
        configured_asset_root = "https://example.invalid/Assets/Isaac/6.1"
        settings.set_string(SETTING_ASSET_ROOT, configured_asset_root)

        try:
            collection = IsaacCollection()
            await collection.populate_children_async()

            self.assertGreater(len(collection.children), 0)
            for child in collection.children.values():
                self.assertTrue(child.path.startswith(f"{configured_asset_root}/Isaac/"))
        finally:
            settings.set_string(SETTING_ASSET_ROOT, original_asset_root)

    async def test_file_importer_dialog_contains_isaac_collection(self) -> None:
        """Test the Isaac Sim collection against the live Kit File Picker API."""
        file_importer = get_file_importer()
        self.assertIsNotNone(file_importer)

        test_directory = Path(__file__).parent.as_posix()
        file_importer.show_window(title="Isaac Sim File Picker Test", filename_url=f"{test_directory}/")

        async def wait_for_isaac_collection() -> IsaacCollection:
            while True:
                await omni.kit.app.get_app().next_update_async()
                dialog = file_importer.get_dialog()
                widget = getattr(dialog, "_widget", None) if dialog else None
                file_picker_api = getattr(widget, "api", None) if widget else None
                if file_picker_api:
                    collection = file_picker_api.view.navigation_model.collections.get("Isaac Sim")
                    if collection:
                        return collection

        try:
            collection = await asyncio.wait_for(wait_for_isaac_collection(), timeout=10.0)
        except asyncio.TimeoutError:
            self.fail(
                "The live File Importer dialog did not register the Isaac Sim collection within 10 seconds; "
                "check the File Picker UI-ready event and dialog API integration."
            )

        self.assertIsInstance(collection, IsaacCollection)
        self.assertEqual(collection.identifier, "Isaac Sim")
        self.assertEqual(collection.title, "Isaac Sim")
        self.assertEqual(collection.path, "Isaac Sim://")

    async def test_child_names_disambiguates_configured_folders(self) -> None:
        """Test that configured folders always receive unique, descriptive names."""
        cases = (
            (
                [
                    "omniverse://localhost/NVIDIA/Assets/Isaac/5.0/Isaac/Robots",
                    "omniverse://studio/Custom/Robots",
                ],
                ["Isaac/Robots", "Custom/Robots"],
            ),
            (
                [
                    "omniverse://localhost/First/Common/Robots",
                    "omniverse://studio/Second/Common/Robots",
                ],
                ["First/Common/Robots", "Second/Common/Robots"],
            ),
            (
                [
                    "https://assets.example.com/Isaac/Robots/",
                    "https://assets.example.com/Custom/Robots/",
                ],
                ["Isaac/Robots", "Custom/Robots"],
            ),
            (
                [
                    "omniverse://localhost/Isaac/Robots",
                    "omniverse://localhost/Isaac/Robots",
                    "omniverse://localhost/Isaac/Robots (2)",
                ],
                ["Robots", "Robots (2)", "Robots (2) (2)"],
            ),
            ([], []),
        )

        for urls, expected_names in cases:
            with self.subTest(urls=urls):
                names = IsaacCollection._child_names(urls)
                self.assertEqual(names, expected_names)
                self.assertEqual(len(names), len(set(names)))
