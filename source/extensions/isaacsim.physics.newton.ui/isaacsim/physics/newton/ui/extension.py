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

"""Newton/Mujoco physics UI extension: schema registration and property widgets."""

import omni.ext
import omni.kit.property.physics as property
import omni.kit.property.usd as usd_property
from omni.physics.isaacsimready import get_capability_manager, get_variant_switcher

from .array_widget import close_all_editors
from .mujoco_schemas import MujocoUiDefinitions, get_mujoco_schema_names
from .mujoco_widgets import MujocoWidgets
from .mujoco_widgets import ignored_schemas as ignored_mujoco_schemas
from .newton_menu import NewtonMenu
from .newton_schemas import NewtonUiDefinitions, get_newton_schema_names
from .newton_widgets import NewtonWidgets
from .newton_widgets import ignored_schemas as ignored_newton_schemas


class PhysicsNewtonUIExtension(omni.ext.IExt):
    """Extension that registers Newton and Mujoco schema names and property UI."""

    def __init__(self) -> None:
        super().__init__()

    def on_startup(self, _ext_id: str) -> None:
        """Register Newton/Mujoco schema names, property widgets, and variant switcher.

        Args:
            _ext_id: Extension identifier from the extension manager.
        """
        capability_manager = get_capability_manager()

        # Register all Mujoco schema names
        prim_types, api_schemas = get_mujoco_schema_names()
        capability_manager.register_schema_type_names(prim_types)
        capability_manager.register_api_schema_names(api_schemas)

        # Register all Newton schema names
        prim_types, api_schemas = get_newton_schema_names()
        capability_manager.register_schema_type_names(prim_types)
        capability_manager.register_api_schema_names(api_schemas)

        # remove the newton schema registered by physx
        property.unregister_parent_schema("newton")
        # Register newton widgets first, this will put them on top of the mujoco widgets
        property.register_parent_schema(
            "newton",
            "Newton",
            NewtonUiDefinitions.widgets,
            NewtonUiDefinitions.property_builders,
            NewtonUiDefinitions.property_order,
            NewtonUiDefinitions.extensions,
            NewtonUiDefinitions.extras,
            NewtonUiDefinitions.ignore,
        )

        # Register mujoco widgets
        property.register_parent_schema(
            "mjcPhysics",
            "Mujoco",
            MujocoUiDefinitions.widgets,
            MujocoUiDefinitions.property_builders,
            MujocoUiDefinitions.property_order,
            MujocoUiDefinitions.extensions,
            MujocoUiDefinitions.extras,
            MujocoUiDefinitions.ignore,
        )

        # The parent-schema manager omits ignored schemas from its private ownership list.
        # These schemas are ignored only because the custom widgets below render them.
        usd_property.register_schema(
            "NewtonCustomPhysicsWidgets",
            sorted(ignored_newton_schemas | ignored_mujoco_schemas),
            options=usd_property.RegisteredSchemaCodes.PRIVATE,
        )

        # Register simulator-to-variant mappings and the parent schema group
        get_variant_switcher().register_simulator_variant("PhysX", "physx")
        get_variant_switcher().register_simulator_variant("Newton", "mujoco")

        # Register group to fit with the simulator name
        property.register_parent_schema_group(
            "Newton",
            ["mjcPhysics", "newton"],
        )

        self._newton_widgets = NewtonWidgets()
        self._newton_widgets.register(property)
        self._mujoco_widgets = MujocoWidgets()
        self._mujoco_widgets.register(property)

        self._menu = NewtonMenu()
        self._menu.on_startup()

    def on_shutdown(self) -> None:
        """Unregister Newton and Mujoco property schema groups and widgets."""
        # Defer destruction of any open array editor windows out of the shutdown call stack.
        close_all_editors()
        # Unregister property widgets
        self._mujoco_widgets.unregister(property)
        self._mujoco_widgets = None
        self._newton_widgets.unregister(property)
        self._newton_widgets = None
        property.unregister_parent_schema_group("Newton")
        property.unregister_parent_schema("newton")
        property.unregister_parent_schema("mjcPhysics")
        # Technically, we probably need to re-register newton schema from physx
        # however, if we assume user will only use newton with this extension enabled, i think it is ok not to.
        self._menu.on_shutdown()
        self._menu = None
