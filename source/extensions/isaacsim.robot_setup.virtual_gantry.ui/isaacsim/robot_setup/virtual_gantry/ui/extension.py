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

"""Kit extension: Create menu, property panel and hotkeys for the virtual gantry.

All physics lives in the backend ``isaacsim.robot_setup.virtual_gantry``
extension; this one only drives it. The manager is resolved lazily through
``get_manager()`` on every hotkey press rather than cached at startup, since
this extension may start before the backend.
"""

from __future__ import annotations

import gc
import logging
from functools import partial
from typing import Any

import omni.ext
import omni.kit.actions.core
import omni.usd
from isaacsim.robot_setup.virtual_gantry import create_virtual_gantry, get_manager
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items

from .widgets.VirtualGantryPropertiesWidget import VirtualGantryPropertiesWidget

_LOGGER = logging.getLogger(__name__)
_CREATE_ACTION = "isaac_create_virtual_gantry"
_WIDGET_NAME = "virtual_gantry"


class VirtualGantryUIExtension(omni.ext.IExt):
    """Registers the Virtual Gantry create-menu, property widget and hotkeys."""

    def on_startup(self, ext_id: str) -> None:
        """Register the menu, property widget and keyboard hotkeys.

        Args:
            ext_id: Identifier of the extension being started.
        """
        self._ext_id = ext_id
        self._ext_name = omni.ext.get_extension_name(ext_id)
        self._hotkey_sub = None

        # Create > Robotics > Virtual Gantry.
        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.register_action(
            self._ext_name,
            _CREATE_ACTION,
            partial(self._on_create_clicked),
            display_name="Create Virtual Gantry",
            description="Create a virtual gantry that suspends a robot during bring-up",
            tag="Create Virtual Gantry",
        )
        self._menu_items = [
            MenuItemDescription(
                "Robotics",
                sub_menu=[
                    MenuItemDescription(
                        name="Virtual Gantry",
                        onclick_action=(self._ext_name, _CREATE_ACTION),
                    )
                ],
            )
        ]
        add_menu_items(self._menu_items, "Create")

        self._register_widget()

        # Keyboard hotkeys: G toggles, [ / ] shorten / lengthen the rope. The
        # viewport must have keyboard focus for these to fire.
        self._hotkey_sub = self._setup_hotkeys()

    def on_shutdown(self) -> None:
        """Tear down the menu, property widget and hotkeys."""
        if self._hotkey_sub is not None:
            try:
                import carb.input
                import omni.appwindow

                input_iface = carb.input.acquire_input_interface()
                keyboard = omni.appwindow.get_default_app_window().get_keyboard()
                input_iface.unsubscribe_to_keyboard_events(keyboard, self._hotkey_sub)
            except Exception:  # noqa: BLE001
                pass
            self._hotkey_sub = None

        try:
            remove_menu_items(self._menu_items, "Create")
        except Exception:  # noqa: BLE001
            pass
        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.deregister_action(self._ext_name, _CREATE_ACTION)

        self._unregister_widget()
        gc.collect()

    # ------------------------------------------------------------------
    def _setup_hotkeys(self) -> Any:
        """Bind G / [ / ] keyboard hotkeys to the backend's manager.

                ``G`` toggles every gantry on/off, ``[`` / ``]`` shorten / lengthen the
                rope by 5 mm. The viewport must have keyboard focus for events to fire.
                Returns the carb subscription handle, or ``None`` when carb input is
                unavailable (headless without an app window).

        Returns:
            The requested value.
        """
        try:
            import carb.input
            import omni.appwindow
        except Exception:  # noqa: BLE001
            return None
        try:
            input_iface = carb.input.acquire_input_interface()
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        except Exception:  # noqa: BLE001
            return None

        key_input = carb.input.KeyboardInput
        press = carb.input.KeyboardEventType.KEY_PRESS
        repeat = carb.input.KeyboardEventType.KEY_REPEAT
        step = 0.005

        def _on_keyboard(event: Any, *_a: Any, **_k: Any) -> bool:
            event_type = event.type
            if event_type not in (press, repeat):
                return True
            # Resolved per press: the backend extension may start after this
            # one, and may be stopped and restarted independently.
            manager = get_manager()
            if manager is None:
                return True
            key = event.input
            if key == key_input.G:
                if event_type == press:  # toggle on press only; repeat would thrash
                    manager.toggle_all()
            elif key == key_input.LEFT_BRACKET:
                manager.adjust_rope_length_all(-step)
            elif key == key_input.RIGHT_BRACKET:
                manager.adjust_rope_length_all(step)
            return True

        try:
            return input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard)
        except Exception:  # noqa: BLE001
            return None

    def _on_create_clicked(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
        hint = paths[0] if paths else ""
        default_prim = stage.GetDefaultPrim()
        parent = str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else "/World"
        create_virtual_gantry(stage, parent_path=parent, wire_hint=hint)

    def _register_widget(self) -> None:
        import omni.kit.window.property as p

        w = p.get_window()
        if w:
            w.register_widget(
                "prim",
                _WIDGET_NAME,
                VirtualGantryPropertiesWidget(title="Virtual Gantry", collapsed=False),
                False,
            )

    def _unregister_widget(self) -> None:
        import omni.kit.window.property as p

        w = p.get_window()
        if w:
            try:
                w.unregister_widget("prim", _WIDGET_NAME)
            except Exception:  # noqa: BLE001
                pass
