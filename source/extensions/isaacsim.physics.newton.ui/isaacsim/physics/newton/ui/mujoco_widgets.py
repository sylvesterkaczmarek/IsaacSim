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

"""Property-panel menu widgets for applying MuJoCo API schemas."""

from collections import namedtuple
from functools import partial

from pxr import Sdf, Usd

from .utils import NewtonWidgetBase, apply_codeless_api_with_dependencies


def _get_schema_attributes(schema: str) -> list[Sdf.AttributeSpec]:
    """Get attribute specifications declared by an applied API schema.

    Args:
        schema: Applied API schema name.

    Returns:
        Attribute specifications declared by the schema.
    """
    schema_definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema)
    if not schema_definition:
        return []
    return [
        property_spec
        for name in schema_definition.GetPropertyNames()
        if isinstance(property_spec := schema_definition.GetSchemaPropertySpec(name), Sdf.AttributeSpec)
    ]


def _can_apply_to_type(prim: Usd.Prim, schema: str, allowed_types: tuple[str, ...]) -> bool:
    """Check whether an API can be applied to the selected prim type.

    Args:
        prim: Prim to inspect.
        schema: API schema that would be applied.
        allowed_types: USD schema types accepted by this API.

    Returns:
        True when the prim has an accepted type and does not already have the API.
    """
    return not prim.HasAPI(schema) and any(prim.IsA(allowed_type) for allowed_type in allowed_types)


MujocoSchemaItem = namedtuple(
    "MujocoSchemaItem",
    "name title menu_label schema attributes dependencies allowed_types exclusive_classes",
)
MujocoSchemaItem.__new__.__defaults__ = (
    (),  # attributes
    (),  # dependencies
    (),  # allowed_types
    None,  # exclusive_classes
)


mujoco_schema_menu_items = [
    MujocoSchemaItem(
        "MujocoPhysicsSceneAPI",
        "MuJoCo Scene",
        "Scene",
        "MjcSceneAPI",
        attributes=_get_schema_attributes("MjcSceneAPI"),
        dependencies=("NewtonSceneAPI",),
        allowed_types=("PhysicsScene",),
    ),
    MujocoSchemaItem(
        "MujocoPhysicsJointAPI",
        "MuJoCo Joint",
        "Joint",
        "MjcJointAPI",
        attributes=_get_schema_attributes("MjcJointAPI"),
        allowed_types=("PhysicsJoint",),
    ),
    MujocoSchemaItem(
        "MujocoPhysicsCollisionAPI",
        "MuJoCo Collider",
        "Collider",
        "MjcCollisionAPI",
        attributes=_get_schema_attributes("MjcCollisionAPI"),
        dependencies=("NewtonCollisionAPI",),
        allowed_types=("Gprim",),
    ),
    MujocoSchemaItem(
        "MujocoPhysicsMeshCollisionAPI",
        "MuJoCo Mesh Collider",
        "Mesh Collider",
        "MjcMeshCollisionAPI",
        attributes=_get_schema_attributes("MjcMeshCollisionAPI"),
        dependencies=("NewtonCollisionAPI", "MjcCollisionAPI", "NewtonMeshCollisionAPI"),
        allowed_types=("Mesh",),
        exclusive_classes=("NewtonSDFCollisionAPI",),
    ),
    MujocoSchemaItem(
        "MujocoPhysicsSiteAPI",
        "MuJoCo Site",
        "Site",
        "MjcSiteAPI",
        attributes=_get_schema_attributes("MjcSiteAPI"),
        dependencies=("NewtonSiteAPI",),
        allowed_types=("Gprim",),
    ),
    MujocoSchemaItem(
        "MujocoPhysicsEqualityConnectAPI",
        "MuJoCo Connect Equality",
        "Equality Connect",
        "MjcEqualityConnectAPI",
        attributes=_get_schema_attributes("MjcEqualityConnectAPI"),
        dependencies=("MjcEqualityAPI",),
        allowed_types=("PhysicsSphericalJoint",),
    ),
    MujocoSchemaItem(
        "MujocoPhysicsEqualityWeldAPI",
        "MuJoCo Weld Equality",
        "Equality Weld",
        "MjcEqualityWeldAPI",
        attributes=_get_schema_attributes("MjcEqualityWeldAPI"),
        dependencies=("MjcEqualityAPI",),
        allowed_types=("PhysicsFixedJoint",),
    ),
    MujocoSchemaItem(
        "MujocoPhysicsEqualityJointAPI",
        "MuJoCo Joint Equality",
        "Equality Joint",
        "MjcEqualityJointAPI",
        attributes=_get_schema_attributes("MjcEqualityJointAPI"),
        dependencies=("NewtonMimicAPI",),
        allowed_types=("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"),
    ),
]

# The custom widgets own display and application for these schemas, so the
# automatically generated parent-schema widgets must skip them.
ignored_schemas = {item.schema for item in mujoco_schema_menu_items}


class MujocoWidgets:
    """Collection of property widgets for applying MuJoCo API schemas.

    Creating an instance builds one widget per entry in :data:`mujoco_schema_menu_items`.
    """

    def __init__(self) -> None:
        self._widgets = {}
        for item in mujoco_schema_menu_items:
            apply_fn = partial(
                apply_codeless_api_with_dependencies,
                dependencies=item.dependencies,
                api=item.schema,
            )
            show_fn = partial(
                _can_apply_to_type,
                schema=item.schema,
                allowed_types=item.allowed_types,
            )
            self._widgets[item.name] = NewtonWidgetBase(
                "Physics/Mujoco",
                item.title,
                item.menu_label,
                item.schema,
                attributes=item.attributes,
                apply_fn=apply_fn,
                show_fn=show_fn,
                exclusive_classes=item.exclusive_classes,
            )

    def register(self, property_manager: object) -> None:
        """Register all MuJoCo widgets with the physics property manager.

        Args:
            property_manager: Physics property manager module.
        """
        for name, widget in self._widgets.items():
            property_manager.register_widget(name, widget)

    def unregister(self, property_manager: object) -> None:
        """Unregister all MuJoCo widgets from the physics property manager.

        Args:
            property_manager: Physics property manager module.
        """
        for name in self._widgets:
            property_manager.unregister_widget(name)
