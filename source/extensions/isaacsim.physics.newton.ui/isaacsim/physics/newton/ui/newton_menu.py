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

"""The Newton submenu when you right-click on an object in the viewport."""

import carb
import omni.kit
from carb.eventdispatcher import get_eventdispatcher
from omni.kit.menu.utils import MenuItemDescription

from . import newton_commands

NEWTON_GLYPH_FILE = "menu_physics.svg"


class NewtonMenu:
    """Newton and MuJoCo entries added to the Create menu and viewport context menus."""

    instance = None

    def __init__(self) -> None:
        self._usd_context = omni.usd.get_context()
        self._selected_prim_paths = []
        self._refresh_menu_fn = {
            "Create": self._refresh_create_menu,
            "Add": self._refresh_additional_menu,
            "Remove": self._refresh_additional_menu,
        }

    def on_startup(self) -> None:
        """Build the menus and start tracking the stage selection."""
        self._menu_items = []
        self._create_menu_item = None

        def on_stage_selection_changed_event() -> None:
            self._selected_prim_paths = self._usd_context.get_selection().get_selected_prim_paths()
            omni.kit.menu.utils.refresh_menu_items("Create")

        usd_context = omni.usd.get_context()
        self._stage_event_sub = get_eventdispatcher().observe_event(
            observer_name="isaacsim.physics.newton.ui:NewtonMenu",
            event_name=usd_context.stage_event_name(omni.usd.StageEventType.SELECTION_CHANGED),
            on_event=lambda _: on_stage_selection_changed_event(),
        )

        self._refresh_create_menu()
        self._refresh_additional_menu()

        NewtonMenu.instance = self

    def on_shutdown(self) -> None:
        """Remove the menus and release the selection subscription."""
        self._menu_items = []
        if self._create_menu_item is not None:
            omni.kit.menu.utils.remove_menu_items([self._create_menu_item], "Create")
            self._create_menu_item = None
        self._stage_event_sub = None
        self._selected_prim_paths = []

        self._refresh_menu_fn.clear()
        self._refresh_menu_fn = None

        self._remove_context_menus()

        NewtonMenu.instance = None

    def _remove_context_menus(self) -> None:
        self._viewport_create_menu = None
        self._stage_create_menu = None

    def _refresh_create_menu(self) -> None:
        create_menu_dict = {
            "name": {
                # Using "Physics" will merge the menu with the physics menu
                "Physics": [
                    {
                        "name": {
                            "Newton": [
                                {
                                    "name": "Newton actuator",
                                    # this works in ContextMenu.add_create_menu
                                    "enabled_fn": [lambda *_: True],
                                    "onclick_fn": lambda *_: self._on_actuator_create_click(),
                                }
                            ],
                        },
                    },
                    {
                        "name": {
                            "Mujoco": [
                                {
                                    "name": "Mujoco actuator",
                                    "enabled_fn": [lambda *_: True],
                                    "onclick_fn": lambda *_: self._on_mujoco_prim_create_click(
                                        "MjcActuator", "Actuator"
                                    ),
                                },
                                {
                                    "name": "Mujoco keyframe",
                                    "enabled_fn": [lambda *_: True],
                                    "onclick_fn": lambda *_: self._on_mujoco_prim_create_click(
                                        "MjcKeyframe", "Keyframe"
                                    ),
                                },
                                {
                                    "name": "Mujoco tendon",
                                    "enabled_fn": [lambda *_: True],
                                    "onclick_fn": lambda *_: self._on_mujoco_prim_create_click("MjcTendon", "Tendon"),
                                },
                            ],
                        },
                    },
                ]
            },
            "glyph": NEWTON_GLYPH_FILE,
        }
        self._viewport_create_menu = omni.kit.context_menu.add_menu(
            create_menu_dict, "CREATE", "omni.kit.viewport.window"
        )
        self._stage_create_menu = omni.kit.context_menu.add_menu(create_menu_dict, "CREATE", "omni.kit.widget.stage")

        # NOTE: rewrite all to actions when the other menus support them too ...
        ext_id = "isaacsim.physics.newton.ui"
        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.deregister_all_actions_for_extension(ext_id)

        def convert_to_action(item: dict) -> None:
            fn = item.get("onclick_fn")
            name = item.get("name")
            if not fn or not name:
                return

            action_registry.register_action(
                ext_id,
                name,
                fn,
                display_name=f"Create->Physics->{name}",
                description=name,
                tag="Physics Menu Actions",
            )

            item["onclick_action"] = (ext_id, name)

        def create_submenu(item: dict) -> MenuItemDescription:
            item_dict = item["name"]
            name = next(iter(item_dict))
            subitems = []
            for subitem in item_dict[name]:
                if isinstance(subitem["name"], dict):
                    subitems.append(create_submenu(subitem))
                else:
                    convert_to_action(subitem)
                    subitems.append(
                        MenuItemDescription(
                            name=subitem.get("name"),
                            enable_fn=subitem.get("enabled_fn"),
                            glyph=subitem.get("glyph"),
                            show_fn=subitem.get("show_fn"),
                            onclick_action=subitem.get("onclick_action"),
                        )
                    )
            return MenuItemDescription(
                name=name,
                enable_fn=item.get("enabled_fn"),
                glyph=item.get("glyph"),
                sub_menu=subitems,
            )

        if self._create_menu_item is not None:
            omni.kit.menu.utils.remove_menu_items([self._create_menu_item], "Create")

        self._create_menu_item = create_submenu(create_menu_dict)
        self._create_menu_item.appear_after = "Deformable"
        omni.kit.menu.utils.add_menu_items([self._create_menu_item], "Create")

    def _refresh_additional_menu(self) -> None: ...

    def _on_actuator_create_click(self) -> None:
        stage = self._usd_context.get_stage()

        selected_prim_paths = self._selected_prim_paths
        selected_len = len(selected_prim_paths)
        if selected_len > 1:
            carb.log_warn("Multiple prims selected. Using the first as parent.")
        if selected_len == 0:
            selected_prim_path = "/"  # use root path as parent
        else:
            selected_prim_path = selected_prim_paths[0]
        ret, new_actuator = newton_commands.CreateNewtonActuatorCommand.execute(stage, selected_prim_path)

        if new_actuator:
            self._usd_context.get_selection().set_selected_prim_paths([new_actuator.GetPath().pathString], True)
        else:
            carb.log_warn("No newton actuator has been created.")

    def _on_mujoco_prim_create_click(self, type_name: str, base_name: str) -> None:
        """Create a concrete MuJoCo typed prim under the current selection (or the stage root).

        Args:
            type_name: Concrete MuJoCo schema type to instantiate.
            base_name: Base prim name, made unique when the path is already taken.
        """
        stage = self._usd_context.get_stage()

        selected_prim_paths = self._selected_prim_paths
        selected_len = len(selected_prim_paths)
        if selected_len > 1:
            carb.log_warn("Multiple prims selected. Using the first as parent.")
        if selected_len == 0:
            selected_prim_path = "/"  # use root path as parent
        else:
            selected_prim_path = selected_prim_paths[0]

        _, new_prim = newton_commands.CreateMujocoPrimCommand.execute(stage, selected_prim_path, type_name, base_name)

        if new_prim:
            self._usd_context.get_selection().set_selected_prim_paths([new_prim.GetPath().pathString], True)
        else:
            carb.log_warn(f"No {type_name} prim has been created.")
