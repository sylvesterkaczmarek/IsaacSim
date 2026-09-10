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

"""Newton schema discovery and UI definitions for the Newton physics extension."""


def _get_newton_schema_names(pluginName: str) -> tuple[set[str], set[str]]:
    """Get all schema type names from the specified plugin.

    Args:
        pluginName: USD plugin name to query for schema types.

    Returns:
        A tuple of (prim_type_names, api_schema_names) where:
            - prim_type_names: Set of typed schema names (e.g., "PhysicsScene", "PhysicsJoint")
            - api_schema_names: Set of API schema names (e.g., "PhysicsRigidBodyAPI", "PhysicsCollisionAPI")
    """
    from pxr import Plug, Tf, Usd

    prim_type_names: set[str] = set()
    api_schema_names: set[str] = set()

    plug_registry = Plug.Registry()
    schema_registry = Usd.SchemaRegistry()

    plugin = plug_registry.GetPluginWithName(pluginName)
    if plugin is None:
        return prim_type_names, api_schema_names

    # Get all types derived from UsdSchemaBase
    all_types = Tf.Type.Find(Usd.SchemaBase).GetAllDerivedTypes()

    for tf_type in all_types:
        if not plugin.DeclaresType(tf_type):
            continue

        # Get the schema name for this type
        schema_type_name = schema_registry.GetSchemaTypeName(tf_type)
        if not schema_type_name:
            continue

        # Get schema kind to determine if it's a prim type or API schema
        schema_kind = schema_registry.GetSchemaKind(tf_type)

        if schema_kind in (Usd.SchemaKind.AbstractTyped, Usd.SchemaKind.ConcreteTyped):
            # This is a typed schema (prim type)
            prim_type_names.add(schema_type_name)
        elif schema_kind in (Usd.SchemaKind.SingleApplyAPI, Usd.SchemaKind.MultipleApplyAPI):
            # This is an API schema
            api_schema_names.add(schema_type_name)

    return prim_type_names, api_schema_names


def get_newton_schema_names() -> tuple[set[str], set[str]]:
    """Get all Newton schema type names.

    Returns:
        A tuple of (prim_type_names, api_schema_names) for Newton schemas.
    """
    prim_type_names: set[str] = set()
    api_schema_names: set[str] = set()

    prim_types, api_schemas = _get_newton_schema_names("newton")
    prim_type_names.update(prim_types)
    api_schema_names.update(api_schemas)

    # List of schemas that are not capabilities
    not_capabilities = [
        "NewtonSceneAPI",
        "NewtonXpbdSceneAPI",
        "NewtonVbdSceneAPI",
        "NewtonCollisionAPI",
        "NewtonMeshCollisionAPI",
        "NewtonMaterialAPI",
    ]

    for schema_name in not_capabilities:
        prim_type_names.discard(schema_name)
        api_schema_names.discard(schema_name)

    return prim_type_names, api_schema_names


# scene widget with hardcoded solver property
import omni.kit.undo
from omni.kit.commands import execute
from omni.kit.property.physics.widgets import ExtensionSchemaWidget, UiProp
from omni.physics.isaacsimready import get_variant_switcher

from .utils import build_remove_schema_frame_header, request_property_window_refresh

_NEWTON_SCENE_SCHEMA = "NewtonSceneAPI"


class ExtendedNewtonSceneWidget(ExtensionSchemaWidget):
    """A specialized widget for Newton physics scene property display and editing.

    This widget extends the ExtensionSchemaWidget to provide custom handling for Newton physics
    scene properties. It filters and adds Newton-specific properties when the active simulation
    variant is set to "Newton", including the Newton solver selection property.

    The widget automatically detects when Newton physics is active and dynamically adds the
    "newton:solver" property to the property panel, allowing users to configure the solver
    used by the Newton physics system.
    """

    def on_remove_schema(self) -> None:
        """Unapply ``NewtonSceneAPI`` from every selected prim that has it."""
        omni.kit.undo.begin_group()
        for prim in self._prims_with_schema():
            execute("UnapplyCodelessAPISchemaCommand", api=_NEWTON_SCENE_SCHEMA, prim=prim)
        omni.kit.undo.end_group()
        request_property_window_refresh()

    def _prims_with_schema(self) -> list:
        """Collect the selected prims that currently have ``NewtonSceneAPI`` applied.

        Returns:
            The prims in the payload with the schema applied.
        """
        if not self._payload:
            return []
        prims = [self._get_prim(prim_path) for prim_path in self._payload]
        return [prim for prim in prims if prim and prim.HasAPI(_NEWTON_SCENE_SCHEMA)]

    def _build_frame_header(self, collapsed: bool, text: str, group_id: str | None = None) -> None:
        """Build the frame header, matching the remove button of the other Newton schemas.

        ``ExtensionSchemaWidget`` keeps the frame visible whenever the parent ``PhysicsScene``
        is present, so the button only appears once the API is applied.

        Args:
            collapsed: Whether the frame is currently collapsed.
            text: The header label text.
            group_id: Optional identifier for the header.
        """
        on_remove = self.on_remove_schema if self._prims_with_schema() else None
        build_remove_schema_frame_header(collapsed, text, _NEWTON_SCENE_SCHEMA, on_remove)

    def _filter_props_to_build(self, prim: "Usd.Prim") -> list:
        """Filters properties to build for Newton scene widgets.

        Adds Newton solver property to the filtered properties when Newton simulation is active.

        Args:
            prim: The USD prim to filter properties for.

        Returns:
            List of filtered properties including Newton solver property when applicable.
        """
        filtered_props = super()._filter_props_to_build(prim)
        if get_variant_switcher().get_active_simulation()[1] == "Newton":
            filtered_props.append(
                UiProp().from_custom("newton:solver", "Solver", "", "token", "mujoco", "Solver used by Newton")
            )
        return filtered_props


# NOTE: omni.physics.physx.ui/omni/physics/physxui/schemas/physxschema.py for reference

from pxr import UsdPhysics

from .utils import DisableByCallbackBuilder, PrimType, ValueChangeByCallbackTokenBuilder, make_hide_cb

CallbackBuilder = DisableByCallbackBuilder

import omni.kit.property.physics
import omni.ui as ui
from isaacsim.physics.newton import newton_solver_to_api_schema as newton_solver_to_api_schema

from .newton_widgets import ignored_schemas


def SolverChangeCallBack(model: ui.AbstractItemModel, dummy: None) -> None:
    new_solver = model.get_value_as_token()
    if not model._valid_stage:
        return
    for path in model._object_paths:
        prim = model._valid_stage.GetPrimAtPath(path.GetPrimPath())
        for solver, api in newton_solver_to_api_schema.items():
            if new_solver == solver:
                if not prim.HasAPI(api):
                    prim.ApplyAPI(api)
            else:
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
    omni.kit.property.physics.refresh()


class NewtonUiDefinitions:
    """UI definitions (widgets, property builders, ordering) for Newton schemas."""

    ignore = ignored_schemas  # ignore schemas that has its own custom widgets
    """Empty dictionary for properties to ignore in the UI."""
    extensions = {
        UsdPhysics.Scene: ["NewtonSceneAPI"],
    }
    """Maps USD schema types to their Newton extension schemas."""
    extras = {}
    """Empty dictionary for additional UI extras."""
    widgets = {
        "NewtonSceneAPI": ExtendedNewtonSceneWidget,
    }
    """Maps schema names to their corresponding UI widget classes."""
    property_builders = {
        "newton:solver": [
            ValueChangeByCallbackTokenBuilder,
            [],
            [
                ("mujoco", "MuJoCo Warp"),
                ("xpbd", "XPBD"),
                # Disabled VBD until schema release
                # ("vbd", "VBD"),
            ],
            SolverChangeCallBack,
        ],
        # Common Newton/Mjc properties - hidden when Mjc provides the value and Newton hasn't authored its version.
        # SCENE
        "newton:maxSolverIterations": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.SCENE, "max_solver_iterations", None),
        ],
        "newton:timeStepsPerSecond": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.SCENE, "time_steps_per_second", 1000),
        ],
        "newton:gravityEnabled": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.SCENE, "gravity_enabled", True),
        ],
        # JOINT
        "newton:armature": [CallbackBuilder, make_hide_cb("newton", PrimType.JOINT, "armature", 0.0)],
        "newton:friction": [CallbackBuilder, make_hide_cb("newton", PrimType.JOINT, "friction", 0.0)],
        "newton:linear:limitStiffness": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.JOINT, "limit_linear_ke", 1e4),
        ],
        "newton:angular:limitStiffness": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.JOINT, "limit_angular_ke", 1e4),
        ],
        "newton:rotX:limitStiffness": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.JOINT, "limit_rotX_ke", 1e4),
        ],
        "newton:rotY:limitStiffness": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.JOINT, "limit_rotY_ke", 1e4),
        ],
        "newton:rotZ:limitStiffness": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.JOINT, "limit_rotZ_ke", 1e4),
        ],
        "newton:linear:limitDamping": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.JOINT, "limit_linear_kd", 1e1),
        ],
        "newton:angular:limitDamping": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.JOINT, "limit_angular_kd", 1e1),
        ],
        "newton:rotX:limitDamping": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.JOINT, "limit_rotX_kd", 1e1),
        ],
        "newton:rotY:limitDamping": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.JOINT, "limit_rotY_kd", 1e1),
        ],
        "newton:rotZ:limitDamping": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.JOINT, "limit_rotZ_kd", 1e1),
        ],
        # SHAPE
        "newton:maxHullVertices": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.SHAPE, "max_hull_vertices", -1),
        ],
        "newton:contactMargin": [CallbackBuilder, make_hide_cb("newton", PrimType.SHAPE, "margin", 0.0)],
        "newton:contactGap": [CallbackBuilder, make_hide_cb("newton", PrimType.SHAPE, "gap", 0.0)],
        # MATERIAL
        "newton:torsionalFriction": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.MATERIAL, "mu_torsional", 0.005),
        ],
        "newton:rollingFriction": [
            CallbackBuilder,
            make_hide_cb("newton", PrimType.MATERIAL, "mu_rolling", 0.0001),
        ],
    }
    """Maps property names to their UI builder configuration including builder class and display options."""
    property_order = {}
    """Empty dictionary for custom property ordering in the UI."""
