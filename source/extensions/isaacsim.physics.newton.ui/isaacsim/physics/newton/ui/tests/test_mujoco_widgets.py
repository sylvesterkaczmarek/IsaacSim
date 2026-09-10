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

"""Tests for MuJoCo API schema apply-menu filters and composition."""

from unittest.mock import patch

from omni.kit.property.usd.usd_property_widget import UsdPropertyUiEntry
from omni.kit.test.async_unittest import AsyncTestCase
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from ..array_widget import _ARRAY_CONSTRAINTS
from ..mujoco_widgets import _can_apply_to_type, ignored_schemas, mujoco_schema_menu_items
from ..utils import NewtonWidgetBase, apply_codeless_api_with_dependencies


class MujocoWidgetTests(AsyncTestCase):
    """Validate MuJoCo schema menu visibility and composed API application."""

    def setUp(self) -> None:
        """Create an in-memory stage with representative physics prim types."""
        self._stage = Usd.Stage.CreateInMemory()
        self._scene = UsdPhysics.Scene.Define(self._stage, "/World/PhysicsScene").GetPrim()
        self._xform = UsdGeom.Xform.Define(self._stage, "/World/Xform").GetPrim()
        self._cube = UsdGeom.Cube.Define(self._stage, "/World/Cube").GetPrim()
        self._mesh = UsdGeom.Mesh.Define(self._stage, "/World/Mesh").GetPrim()
        self._fixed_joint = UsdPhysics.FixedJoint.Define(self._stage, "/World/FixedJoint").GetPrim()
        self._spherical_joint = UsdPhysics.SphericalJoint.Define(self._stage, "/World/SphericalJoint").GetPrim()
        self._revolute_joint = UsdPhysics.RevoluteJoint.Define(self._stage, "/World/RevoluteJoint").GetPrim()
        self._prismatic_joint = UsdPhysics.PrismaticJoint.Define(self._stage, "/World/PrismaticJoint").GetPrim()

    def tearDown(self) -> None:
        """Release the in-memory stage."""
        self._stage = None

    def test_scene_filter_accepts_only_physics_scenes(self) -> None:
        """Show `MjcSceneAPI` only for physics scene prims."""
        self.assertTrue(_can_apply_to_type(self._scene, "MjcSceneAPI", ("PhysicsScene",)))
        self.assertFalse(_can_apply_to_type(self._xform, "MjcSceneAPI", ("PhysicsScene",)))

    def test_shape_filters_distinguish_gprims_and_meshes(self) -> None:
        """Restrict collider and mesh-collider entries to compatible geometry."""
        self.assertTrue(_can_apply_to_type(self._cube, "MjcCollisionAPI", ("Gprim",)))
        self.assertTrue(_can_apply_to_type(self._mesh, "MjcCollisionAPI", ("Gprim",)))
        self.assertFalse(_can_apply_to_type(self._xform, "MjcCollisionAPI", ("Gprim",)))
        self.assertTrue(_can_apply_to_type(self._mesh, "MjcMeshCollisionAPI", ("Mesh",)))
        self.assertFalse(_can_apply_to_type(self._cube, "MjcMeshCollisionAPI", ("Mesh",)))

    def test_equality_filters_match_supported_joint_types(self) -> None:
        """Restrict equality schemas to the joint types consumed by Newton."""
        self.assertTrue(_can_apply_to_type(self._spherical_joint, "MjcEqualityConnectAPI", ("PhysicsSphericalJoint",)))
        self.assertFalse(_can_apply_to_type(self._fixed_joint, "MjcEqualityConnectAPI", ("PhysicsSphericalJoint",)))
        self.assertTrue(_can_apply_to_type(self._fixed_joint, "MjcEqualityWeldAPI", ("PhysicsFixedJoint",)))
        scalar_joint_types = ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
        self.assertTrue(_can_apply_to_type(self._revolute_joint, "MjcEqualityJointAPI", scalar_joint_types))
        self.assertTrue(_can_apply_to_type(self._prismatic_joint, "MjcEqualityJointAPI", scalar_joint_types))
        self.assertFalse(_can_apply_to_type(self._spherical_joint, "MjcEqualityJointAPI", scalar_joint_types))

    def test_filter_hides_already_applied_schema(self) -> None:
        """Hide an Apply entry after its schema has been applied."""
        self._scene.ApplyAPI("MjcSceneAPI")
        self.assertFalse(_can_apply_to_type(self._scene, "MjcSceneAPI", ("PhysicsScene",)))

    def test_composed_scene_application(self) -> None:
        """Apply `NewtonSceneAPI` before `MjcSceneAPI`."""
        with patch(
            "isaacsim.physics.newton.ui.utils.execute",
            side_effect=lambda _command, api, prim: prim.ApplyAPI(api),
        ) as execute_mock:
            apply_codeless_api_with_dependencies(self._scene, ("NewtonSceneAPI",), "MjcSceneAPI")
        self.assertTrue(self._scene.HasAPI("NewtonSceneAPI"))
        self.assertTrue(self._scene.HasAPI("MjcSceneAPI"))
        self.assertEqual(
            [call.kwargs["api"] for call in execute_mock.call_args_list], ["NewtonSceneAPI", "MjcSceneAPI"]
        )

    def test_composed_mesh_collider_application(self) -> None:
        """Apply the collision API chain required by a MuJoCo mesh collider."""
        dependencies = ("NewtonCollisionAPI", "MjcCollisionAPI", "NewtonMeshCollisionAPI")
        with patch(
            "isaacsim.physics.newton.ui.utils.execute",
            side_effect=lambda _command, api, prim: prim.ApplyAPI(api),
        ) as execute_mock:
            apply_codeless_api_with_dependencies(self._mesh, dependencies, "MjcMeshCollisionAPI")
        for schema in (*dependencies, "MjcMeshCollisionAPI"):
            self.assertTrue(self._mesh.HasAPI(schema))
        self.assertEqual(
            [call.kwargs["api"] for call in execute_mock.call_args_list],
            [*dependencies, "MjcMeshCollisionAPI"],
        )

    def test_custom_widgets_replace_automatic_widgets(self) -> None:
        """Ignore every custom Apply widget schema in automatic parent-schema rendering."""
        self.assertEqual(ignored_schemas, {item.schema for item in mujoco_schema_menu_items})

    def test_custom_widget_layout_routes_arrays_to_editor(self) -> None:
        """Attach registered array and resolver builders inside custom Apply widgets.

        These schemas are in ``ignored_schemas``, so no automatic ``PhysicsWidget``
        consumes the registered builders. The custom widget must attach them itself.
        """
        widget = NewtonWidgetBase("Physics/Mujoco", "MuJoCo Joint", "Joint", "MjcJointAPI")
        try:
            array_entry = UsdPropertyUiEntry(
                "mjc:solreflimit",
                "",
                {Sdf.PrimSpec.TypeNameKey: "double[]"},
                Usd.Attribute,
            )
            scalar_entry = UsdPropertyUiEntry(
                "mjc:armature",
                "",
                {Sdf.PrimSpec.TypeNameKey: "double"},
                Usd.Attribute,
            )
            hidden_entry = UsdPropertyUiEntry(
                "mjc:compiler:useThread",
                "",
                {Sdf.PrimSpec.TypeNameKey: "bool"},
                Usd.Attribute,
            )
            widget._customize_props_layout([array_entry, scalar_entry, hidden_entry])
            self.assertIsNotNone(array_entry.build_fn)
            self.assertIsNotNone(scalar_entry.build_fn)
            self.assertIsNotNone(hidden_entry.build_fn)
        finally:
            widget.clean()

    def test_every_constrained_array_is_reachable_from_a_render_path(self) -> None:
        """Ensure each size-constrained array renders through the pop-up editor.

        An attribute is reachable either through the physics ``property_builders`` map
        (typed prims such as ``MjcActuator``) or through a custom widget whose schema owns
        it (applied APIs such as ``MjcJointAPI``).
        """
        from ..mujoco_schemas import ARRAY_EDITOR, MujocoUiDefinitions

        widget_owned = {
            attr.name for item in mujoco_schema_menu_items for attr in item.attributes if attr.name.startswith("mjc:")
        }
        for attr_name in _ARRAY_CONSTRAINTS:
            wired = MujocoUiDefinitions.property_builders.get(attr_name)
            reachable = (wired is not None and ARRAY_EDITOR[0] in wired) or attr_name in widget_owned
            self.assertTrue(reachable, f"{attr_name} has a size constraint but no array-editor render path")
