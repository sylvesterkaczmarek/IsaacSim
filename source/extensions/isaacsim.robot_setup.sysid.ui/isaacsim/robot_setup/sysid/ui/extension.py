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

"""Interactive extension entry point for System Identification."""

from __future__ import annotations

import asyncio
import gc
from typing import Any

import carb
import omni
import omni.kit.actions.core
import omni.kit.app
import omni.timeline
import omni.ui as ui
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items

from .ui_builder import UIBuilder

EXTENSION_TITLE = "System Identification"


class Extension(omni.ext.IExt):
    """System Identification tool under Tools > Robotics > Asset Editors."""

    def on_startup(self, ext_id: str) -> None:
        """Handle startup.

        Args:
            ext_id: Ext id value.
        """
        self.ext_id = ext_id
        self._ext_name = omni.ext.get_extension_name(ext_id)
        self._dock_task = None
        self._menu_items = []
        self._timeline_event_subscriptions = []
        self._window = ui.Window(
            title=EXTENSION_TITLE,
            width=720,
            height=760,
            visible=False,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        self._window.set_visibility_changed_fn(self._on_window)
        self._ui_builder = None

        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.register_action(
            self._ext_name,
            f"CreateUIExtension:{EXTENSION_TITLE}",
            self._menu_callback,
            description=f"Open {EXTENSION_TITLE}",
        )
        items = [
            MenuItemDescription(
                name=EXTENSION_TITLE,
                onclick_action=(self._ext_name, f"CreateUIExtension:{EXTENSION_TITLE}"),
            )
        ]
        asset_editors = [MenuItemDescription(name="Asset Editors", sub_menu=items)]
        self._menu_items = [MenuItemDescription(name="Robotics", sub_menu=asset_editors)]
        add_menu_items(self._menu_items, "Tools")

    def on_shutdown(self) -> None:
        """Handle shutdown."""
        self._timeline_event_subscriptions = []
        if getattr(self, "_menu_items", None):
            remove_menu_items(self._menu_items, "Tools")
        ext_name = getattr(self, "_ext_name", "")
        if ext_name:
            try:
                omni.kit.actions.core.get_action_registry().deregister_action(
                    ext_name, f"CreateUIExtension:{EXTENSION_TITLE}"
                )
            except (KeyError, RuntimeError) as exc:
                carb.log_warn(f"SysId UI: failed to deregister action during shutdown: {exc}")
        if getattr(self, "_dock_task", None) is not None:
            self._dock_task.cancel()
            self._dock_task = None
        if getattr(self, "_ui_builder", None) is not None:
            self._ui_builder.cleanup()
            self._ui_builder = None
        self._window = None
        gc.collect()

    def _menu_callback(self) -> None:
        if getattr(self, "_window", None):
            self._window.visible = True
            if hasattr(self._window, "focus"):
                self._window.focus()
        if getattr(self, "_ui_builder", None) is not None:
            self._ui_builder.on_menu_callback()

    def _on_window(self, visible: bool) -> None:
        if visible:
            self._build_ui()
            self._subscribe_timeline_events()
        elif getattr(self, "_ui_builder", None) is not None:
            self._timeline_event_subscriptions = []
            self._ui_builder.on_window_hidden()
            self._ui_builder = None

    def _subscribe_timeline_events(self) -> None:
        if self._timeline_event_subscriptions:
            return
        event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._timeline_event_subscriptions = [
            event_stream.create_subscription_to_pop_by_type(
                int(event_type),
                self._on_timeline_event,
                name=f"isaacsim.robot_setup.sysid.ui.{event_type.name.lower()}",
            )
            for event_type in (
                omni.timeline.TimelineEventType.PLAY,
                omni.timeline.TimelineEventType.PAUSE,
                omni.timeline.TimelineEventType.STOP,
            )
        ]

    def _on_timeline_event(self, event: object) -> None:
        if getattr(self, "_ui_builder", None) is not None:
            self._ui_builder.on_timeline_event(event)

    def _build_ui(self) -> None:
        if getattr(self, "_ui_builder", None) is not None:
            self._ui_builder.cleanup()
        self._ui_builder = UIBuilder()
        with self._window.frame:
            self._ui_builder.build_ui()

        async def dock_window() -> None:
            """Handle dock window."""
            await omni.kit.app.get_app().next_update_async()

            def dock(space: Any, name: str, location: Any, pos: float = 0.5) -> Any:
                """Handle dock.

                Args:
                    space: Space value.
                    name: Name value.
                    location: Location value.
                    pos: Pos value.

                Returns:
                    The resulting value.
                """
                window = omni.ui.Workspace.get_window(name)
                if window and space:
                    window.dock_in(space, location, pos)
                return window

            dock(ui.Workspace.get_window("Viewport"), EXTENSION_TITLE, omni.ui.DockPosition.LEFT, 0.55)
            await omni.kit.app.get_app().next_update_async()

        if getattr(self, "_dock_task", None) is not None:
            self._dock_task.cancel()
        self._dock_task = asyncio.ensure_future(dock_window())
