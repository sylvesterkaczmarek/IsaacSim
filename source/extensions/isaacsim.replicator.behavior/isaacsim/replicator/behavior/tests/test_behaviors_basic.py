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

"""Basic functionality tests for behavior scripts."""

from __future__ import annotations

import importlib
import os
from unittest.mock import AsyncMock, patch

import carb
import isaacsim.core.experimental.utils.physics as physics_utils
import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.xform as xform_utils
import isaacsim.replicator.behavior.behaviors as behaviors_module
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.kit.test
import omni.timeline
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.replicator.behavior.behaviors import (
    LightRandomizer,
    LocationRandomizer,
    LookAtBehavior,
    RotationRandomizer,
    TextureRandomizer,
    VolumeStackRandomizer,
)
from isaacsim.replicator.behavior.behaviors.volume_stack_randomizer import BehaviorState
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from isaacsim.replicator.behavior.utils.behavior_utils import (
    add_behavior_script,
    csv_has_relative_asset_url,
    order_range_vectors,
    order_scalar_range,
    resolve_behavior_rng,
    resolve_csv_asset_urls,
)
from omni.behavior.scripting.core import BehaviorScript
from pxr import Gf, Sdf

SCRIPTS_ATTR = "omni:scripting:scripts"


class TestBehaviorsBasic(omni.kit.test.AsyncTestCase):
    """Test the basic functionality of the behavior scripts."""

    async def setUp(self) -> None:
        """Set up a new stage before each test."""
        await omni.kit.app.get_app().next_update_async()
        stage_utils.create_new_stage()
        await omni.kit.app.get_app().next_update_async()

    async def tearDown(self) -> None:
        """Close the stage after each test."""
        await omni.kit.app.get_app().next_update_async()
        stage_utils.close_stage()
        await omni.kit.app.get_app().next_update_async()

    async def check_exposed_variables(self, behavior_class: type) -> None:
        """Verify exposed variables are correctly created and removed for a behavior class.

        Args:
            behavior_class: Behavior class whose exposed variables should be checked.
        """
        # Make sure behavior_class is of type BehaviorScript
        self.assertTrue(
            issubclass(behavior_class, BehaviorScript),
            f"Behavior class '{behavior_class.__name__}' is not a subclass of BehaviorScript",
        )

        # Create a fresh stage with a root prim and some child prims
        await stage_utils.create_new_stage_async()
        root_prim_path = "/World/RootPrim"
        root_prim = stage_utils.define_prim(root_prim_path, "Xform")

        # Add scripting API to the prim.
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[root_prim_path])
        await omni.kit.app.get_app().next_update_async()
        scripts_attr = root_prim.GetAttribute(SCRIPTS_ATTR)
        self.assertTrue(
            scripts_attr and scripts_attr.IsValid(), f"No '{SCRIPTS_ATTR}' attribute found on prim: {root_prim_path}"
        )

        # Get the script path from the behavior class's module
        module = importlib.import_module(behavior_class.__module__)
        script_full_path = module.__file__

        self.assertTrue(os.path.exists(script_full_path), f"Script path does not exist: {script_full_path}")
        self.assertTrue(script_full_path.endswith(".py"), f"Script path is not a .py file: {script_full_path}")

        add_behavior_script(root_prim, script_full_path)

        # NOTE, at least 3 updates are needed to ensure the script is loaded and the exposed vars are set
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

        # Check if the class has VARIABLES_TO_EXPOSE
        self.assertTrue(
            hasattr(behavior_class, "VARIABLES_TO_EXPOSE"),
            f"Behavior class '{behavior_class.__name__}' does not have a 'VARIABLES_TO_EXPOSE' attribute",
        )

        # Check if the class has a BEHAVIOR_NS attribute
        self.assertTrue(
            hasattr(behavior_class, "BEHAVIOR_NS"),
            f"Behavior class '{behavior_class.__name__}' does not have a 'BEHAVIOR_NS' attribute",
        )
        # Full namespace for the exposed variables
        full_ns = f"{EXPOSED_ATTR_NS}:{behavior_class.BEHAVIOR_NS}"

        # Iterate over the variables which should be exposed with the behavior script initialization
        for exposed_var in behavior_class.VARIABLES_TO_EXPOSE:
            # Check if the exposed variable is set on the prim
            attr_name = exposed_var["attr_name"]
            print(f"\tChecking exposed variable: {attr_name}")
            attr_full_name = f"{full_ns}:{attr_name}"
            attribute = root_prim.GetAttribute(attr_full_name)
            self.assertTrue(
                attribute and attribute.IsValid(), f"Attribute '{attr_full_name}' not found on prim: {root_prim_path}"
            )

            # Check if the exposed variable has the correct type
            attr_type = exposed_var["attr_type"]
            self.assertEqual(
                attribute.GetTypeName(),
                attr_type,
                f"Attribute '{attr_full_name}' has incorrect type: {attribute.GetTypeName()} instead of {attr_type}",
            )

            # Check if the exposed variable has a default value
            # NOTE: only checking if there is a default value, not the actual value due to rounding issues
            self.assertTrue(
                attribute.Get() is not None,
                f"Attribute '{attr_full_name}' does not have a valid value: {attribute.Get()} ",
            )

            # Check if the exposed variable has the correct documentation
            doc = exposed_var.get("doc", "")
            self.assertEqual(
                attribute.GetDocumentation(),
                doc,
                f"Attribute '{attr_full_name}' has incorrect documentation: {attribute.GetDocumentation()} instead of {doc}",
            )

        # Basic test to check if the behavior can run for a few frames
        timeline = omni.timeline.get_timeline_interface()
        print(f"\tStarting timeline for several frames")
        timeline.play()
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        print(f"\tPausing timeline")
        timeline.pause()
        await omni.kit.app.get_app().next_update_async()
        print(f"\tStopping timeline")
        timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        # Remove the behavior script from the prim, this should also remove/invalidate the attributes of the exposed variables
        print(f"\tChecking if exposed variables are removed after clearing the scripts attribute")
        scripts_attr.Set([])

        # NOTE, at least 3 updates are needed to ensure the script is loaded and the exposed vars are set
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

        # Check if the exposed variables are removed
        for exposed_var in behavior_class.VARIABLES_TO_EXPOSE:
            attr_name = exposed_var["attr_name"]
            attr_full_name = f"{full_ns}:{attr_name}"
            attribute = root_prim.GetAttribute(attr_full_name)
            self.assertFalse(
                attribute and attribute.IsValid(),
                f"Attribute '{attr_full_name}' with {attribute.Get()} not removed from prim: {root_prim_path}",
            )

    async def test_exposed_variables(self) -> None:
        """Test that all behavior classes correctly expose and remove their variables."""
        # Get the behavior classes to test from the behaviors module __all__ list
        BEHAVIOR_CLASSES = [getattr(behaviors_module, class_name) for class_name in behaviors_module.__all__]
        carb.log_info(f"BEHAVIOR_CLASSES: {BEHAVIOR_CLASSES}")
        for behavior_class in BEHAVIOR_CLASSES:
            print(f"Testing behavior: {behavior_class.__name__}")
            await self.check_exposed_variables(behavior_class)

    async def test_light_randomizer_reset_skips_cached_none_values(self) -> None:
        """Test that light reset skips unset cached attributes and restores remaining prims."""
        stage = stage_utils.get_current_stage()

        unset_prim = stage_utils.define_prim("/World/UnsetLightInputs", "Xform")
        prim_utils.create_prim_attribute(unset_prim, name="inputs:intensity", type_name=Sdf.ValueTypeNames.Float)
        prim_utils.create_prim_attribute(unset_prim, name="inputs:color", type_name=Sdf.ValueTypeNames.Color3f)

        authored_prim = stage_utils.define_prim("/World/AuthoredLightInputs", "Xform")
        prim_utils.create_prim_attribute(authored_prim, name="inputs:intensity", type_name=Sdf.ValueTypeNames.Float)
        prim_utils.create_prim_attribute(authored_prim, name="inputs:color", type_name=Sdf.ValueTypeNames.Color3f)
        initial_intensity = 500.0
        initial_color = Gf.Vec3f(0.1, 0.2, 0.3)
        prim_utils.set_prim_attribute_value(authored_prim, "inputs:intensity", initial_intensity)
        prim_utils.set_prim_attribute_value(authored_prim, "inputs:color", initial_color)

        randomizer = LightRandomizer.__new__(LightRandomizer)
        randomizer._initial_attributes = {}
        randomizer._cache_initial_attributes(unset_prim)
        randomizer._cache_initial_attributes(authored_prim)
        self.assertIsNone(randomizer._initial_attributes[unset_prim]["inputs:intensity"])
        self.assertIsNone(randomizer._initial_attributes[unset_prim]["inputs:color"])

        prim_utils.set_prim_attribute_value(unset_prim, "inputs:intensity", 1000.0)
        prim_utils.set_prim_attribute_value(unset_prim, "inputs:color", Gf.Vec3f(0.5, 0.5, 0.5))
        prim_utils.set_prim_attribute_value(authored_prim, "inputs:intensity", 1000.0)
        prim_utils.set_prim_attribute_value(authored_prim, "inputs:color", Gf.Vec3f(0.5, 0.5, 0.5))

        randomizer._valid_prims = [unset_prim, authored_prim]
        randomizer._update_counter = 1
        randomizer._rng = object()

        randomizer._reset()

        self.assertEqual(prim_utils.get_prim_attribute_value(authored_prim, "inputs:intensity"), initial_intensity)
        self.assertEqual(prim_utils.get_prim_attribute_value(authored_prim, "inputs:color"), initial_color)
        self.assertEqual(prim_utils.get_prim_attribute_value(unset_prim, "inputs:intensity"), 1000.0)
        self.assertEqual(prim_utils.get_prim_attribute_value(unset_prim, "inputs:color"), Gf.Vec3f(0.5, 0.5, 0.5))
        self.assertEqual(randomizer._valid_prims, [])
        self.assertEqual(randomizer._initial_attributes, {})
        self.assertEqual(randomizer._update_counter, 0)
        self.assertIsNone(randomizer._rng)

    async def test_location_randomizer_uses_functional_transform_and_restores_pose(self) -> None:
        """Test location randomization and restoration through the functional transform API."""
        stage = stage_utils.get_current_stage()
        prim = stage_utils.define_prim("/World/LocationTarget", "Xform")
        XformPrim("/World/LocationTarget", reset_xform_op_properties=True).set_local_poses(
            translations=[[1.0, 2.0, 3.0]]
        )

        randomizer = LocationRandomizer.__new__(LocationRandomizer)
        randomizer._valid_prims = [prim]
        randomizer._initial_locations = {prim: Gf.Vec3d(1.0, 2.0, 3.0)}
        randomizer._target_offsets = {}
        randomizer._target_prim = None
        randomizer._use_relative_frame = True
        randomizer._min_position = Gf.Vec3d(0.5, -0.25, 1.0)
        randomizer._max_position = Gf.Vec3d(0.5, -0.25, 1.0)
        randomizer._interval = 1
        randomizer._update_counter = 1
        randomizer._rng = np.random.default_rng(7)

        with patch.object(LocationRandomizer, "stage", stage, create=True):
            randomizer._randomize_location(prim)
            translation, _ = xform_utils.get_local_pose(prim, device="cpu")
            np.testing.assert_allclose(translation.numpy(), [1.5, 1.75, 4.0])

            randomizer._reset()
            translation, _ = xform_utils.get_local_pose(prim, device="cpu")
            np.testing.assert_allclose(translation.numpy(), [1.0, 2.0, 3.0])

    async def test_location_randomizer_skips_invalid_cached_prims(self) -> None:
        """Test that deleted prim handles are ignored during application and reset."""
        stage = stage_utils.get_current_stage()
        prim = stage_utils.define_prim("/World/DeletedLocationTarget", "Xform")

        randomizer = LocationRandomizer.__new__(LocationRandomizer)
        randomizer._valid_prims = [prim]
        randomizer._initial_locations = {prim: Gf.Vec3d(0.0)}
        randomizer._target_offsets = {}
        randomizer._target_prim = None
        randomizer._use_relative_frame = True
        randomizer._min_position = Gf.Vec3d(0.0)
        randomizer._max_position = Gf.Vec3d(0.0)
        randomizer._interval = 1
        randomizer._update_counter = 1
        randomizer._rng = np.random.default_rng(7)

        stage_utils.delete_prim(prim)
        with patch.object(LocationRandomizer, "stage", stage, create=True):
            randomizer._apply_behavior()
            randomizer._reset()

        self.assertEqual(randomizer._valid_prims, [])

    async def test_rotation_randomizer_uses_functional_transform_and_restores_pose(self) -> None:
        """Test rotation randomization and restoration through the functional transform API."""
        stage = stage_utils.get_current_stage()
        prim = stage_utils.define_prim("/World/RotationTarget", "Xform")
        initial_quaternion = Gf.Rotation(Gf.Vec3d.ZAxis(), 15.0).GetQuat()
        initial_orientation = [initial_quaternion.GetReal(), *initial_quaternion.GetImaginary()]
        XformPrim("/World/RotationTarget", reset_xform_op_properties=True).set_local_poses(
            orientations=[initial_orientation]
        )

        randomizer = RotationRandomizer.__new__(RotationRandomizer)
        randomizer._valid_prims = [prim]
        randomizer._initial_rotations = {prim: initial_orientation}
        randomizer._min_rotation = Gf.Vec3d(30.0, 20.0, 10.0)
        randomizer._max_rotation = Gf.Vec3d(30.0, 20.0, 10.0)
        randomizer._interval = 1
        randomizer._update_counter = 1
        randomizer._rng = np.random.default_rng(7)

        with patch.object(RotationRandomizer, "stage", stage, create=True):
            randomizer._randomize_rotation(prim)
            _, randomized_orientation = xform_utils.get_local_pose(prim, device="cpu")
            self.assertLess(abs(float(np.dot(randomized_orientation.numpy(), initial_orientation))), 0.999)

            randomizer._reset()
            _, restored_orientation = xform_utils.get_local_pose(prim, device="cpu")
            self.assertAlmostEqual(abs(float(np.dot(restored_orientation.numpy(), initial_orientation))), 1.0, places=5)

    async def test_look_at_behavior_uses_functional_transform_and_restores_pose(self) -> None:
        """Test look-at authoring and restoration through the functional transform API."""
        stage = stage_utils.get_current_stage()
        prim = stage_utils.define_prim("/World/LookAtTarget", "Xform")
        initial_orientation = [1.0, 0.0, 0.0, 0.0]
        XformPrim("/World/LookAtTarget", reset_xform_op_properties=True).set_local_poses(
            translations=[[2.0, 0.0, 0.0]], orientations=[initial_orientation]
        )

        behavior = LookAtBehavior.__new__(LookAtBehavior)
        behavior._valid_prims = [prim]
        behavior._initial_rotations = {prim: initial_orientation}
        behavior._target_prim = None
        behavior._target_location = Gf.Vec3d(0.0, 0.0, 0.0)
        behavior._up_axis = Gf.Vec3d(0.0, 0.0, 1.0)
        behavior._interval = 1
        behavior._update_counter = 1

        with patch.object(LookAtBehavior, "stage", stage, create=True):
            behavior._apply_behavior()
            _, look_at_orientation = xform_utils.get_local_pose(prim, device="cpu")
            self.assertLess(abs(float(np.dot(look_at_orientation.numpy(), initial_orientation))), 0.999)

            behavior._reset()
            _, restored_orientation = xform_utils.get_local_pose(prim, device="cpu")
            self.assertAlmostEqual(abs(float(np.dot(restored_orientation.numpy(), initial_orientation))), 1.0, places=5)

    async def test_look_at_behavior_skips_coincident_target(self) -> None:
        """Test that a coincident look-at target does not raise or alter the prim."""
        stage = stage_utils.get_current_stage()
        prim = stage_utils.define_prim("/World/CoincidentCamera", "Camera")
        behavior = LookAtBehavior.__new__(LookAtBehavior)
        behavior._valid_prims = [prim]
        behavior._target_prim = None
        behavior._target_location = Gf.Vec3d(0.0)
        behavior._up_axis = Gf.Vec3d(0.0, 0.0, 1.0)

        _, initial_orientation = xform_utils.get_world_pose(prim, device="cpu")
        with patch.object(LookAtBehavior, "stage", stage, create=True):
            behavior._apply_behavior()
        _, final_orientation = xform_utils.get_world_pose(prim, device="cpu")

        self.assertTrue(np.allclose(final_orientation.numpy(), initial_orientation.numpy(), atol=1e-5))

    async def test_texture_randomizer_apply_skips_when_no_textures_configured(self) -> None:
        """Test that ``_apply_behavior`` is a no-op when no texture URLs are configured.

        Regression test for the empty-list crash where ``numpy.random.Generator.choice``
        raised on an empty texture list. The guard must log a warning and return early
        without iterating ``_texture_materials`` or touching ``_rng``.
        """
        randomizer = TextureRandomizer.__new__(TextureRandomizer)
        randomizer._texture_urls = []

        # If the guard regresses, the loop below would iterate ``_texture_materials`` and
        # call ``self._rng.choice(...)``; ``_rng = None`` would surface that as AttributeError.
        sentinel_material = object()
        randomizer._texture_materials = [sentinel_material]
        randomizer._rng = None

        # ``prim_path`` is provided by the ``BehaviorScript`` base class at runtime;
        # patch it at the class level so the warning's f-string is well-formed.
        with patch.object(TextureRandomizer, "prim_path", "/World/UnusedRandomizerPrim", create=True):
            randomizer._apply_behavior()

        self.assertEqual(randomizer._texture_urls, [])
        self.assertEqual(randomizer._texture_materials, [sentinel_material])
        self.assertIsNone(randomizer._rng)

    async def test_texture_randomizer_apply_uses_material_wrapper_inputs(self) -> None:
        """Test that texture randomization writes through the material wrapper API."""

        class MockMaterial:
            def __init__(self) -> None:
                self.inputs = {}

            def set_input_values(self, name: str, values: list) -> None:
                self.inputs[name] = values

        material = MockMaterial()
        randomizer = TextureRandomizer.__new__(TextureRandomizer)
        randomizer._texture_urls = ["omniverse://textures/albedo_a.png"]
        randomizer._texture_materials = [material]
        randomizer._rng = np.random.default_rng(7)
        randomizer._project_uvw_probability = 1.0
        randomizer._texture_scale_range = Gf.Vec2f(0.25, 0.25)
        randomizer._texture_rotate_range = Gf.Vec2f(15.0, 15.0)

        randomizer._apply_behavior()

        self.assertEqual(material.inputs["diffuse_texture"], ["omniverse://textures/albedo_a.png"])
        self.assertEqual(material.inputs["project_uvw"], [True])
        self.assertEqual(material.inputs["texture_scale"], [0.25, 0.25])
        self.assertEqual(material.inputs["texture_rotate"], [15.0])

    async def test_volume_stack_randomizer_toggles_core_physics_state(self) -> None:
        """Test that volume stack physics toggles operate through core prim wrappers."""
        stage = stage_utils.get_current_stage()
        root_prim = stage_utils.define_prim("/World/AssetRoot", "Xform")
        cube_prim = Cube("/World/AssetRoot/Cube").prims[0]

        with stage_utils.use_stage(stage):
            physics_utils.apply_rigid_body(root_prim, approximation="convexHull")

        randomizer = VolumeStackRandomizer.__new__(VolumeStackRandomizer)
        with patch.object(VolumeStackRandomizer, "stage", stage, create=True):
            randomizer._set_enabled_collisions(root_prim, False)
            self.assertFalse(prim_utils.get_prim_attribute_value(cube_prim, "physics:collisionEnabled"))

            randomizer._set_enabled_collisions(root_prim, True)
            self.assertTrue(prim_utils.get_prim_attribute_value(cube_prim, "physics:collisionEnabled"))

            randomizer._set_enabled_rigid_body_dynamics(root_prim, False)
            self.assertFalse(prim_utils.get_prim_attribute_value(root_prim, "physics:rigidBodyEnabled"))

            randomizer._set_enabled_rigid_body_dynamics(root_prim, True)
            self.assertTrue(prim_utils.get_prim_attribute_value(root_prim, "physics:rigidBodyEnabled"))

    async def test_volume_stack_randomizer_applies_asset_scale_to_drop_margin(self) -> None:
        """Test that scaled asset bounds reduce the available drop area."""
        stage = stage_utils.get_current_stage()
        surface = Cube("/World/Surface", sizes=[1.0], scales=[[10.0, 10.0, 1.0]]).prims[0]
        asset = Cube("/World/Asset", sizes=[1.0], scales=[[2.0, 3.0, 4.0]]).prims[0]

        randomizer = VolumeStackRandomizer.__new__(VolumeStackRandomizer)
        randomizer._rng = np.random.default_rng(7)
        randomizer._reset_requested = False
        randomizer._physx_dt = 1.0 / 60.0
        randomizer._render_simulation = False

        expected_rng = np.random.default_rng(7)
        expected_position = [expected_rng.uniform(-3.0, 3.0), expected_rng.uniform(-3.0, 3.0), 1.5]
        with patch.object(VolumeStackRandomizer, "stage", stage, create=True):
            await randomizer._start_batched_asset_drop_async([(surface, asset)], drop_height=1.0, sim_steps=0)

        translation, _ = xform_utils.get_local_pose(asset, device="cpu")
        np.testing.assert_allclose(translation.numpy(), expected_position, atol=1e-5)

    async def test_volume_stack_randomizer_clamps_oversized_asset_drop_area(self) -> None:
        """Place oversized assets at the surface center instead of raising on a negative range."""
        stage = stage_utils.get_current_stage()
        surface = Cube("/World/SmallSurface", sizes=[1.0], scales=[[1.0, 1.0, 1.0]]).prims[0]
        asset = Cube("/World/LargeAsset", sizes=[1.0], scales=[[4.0, 4.0, 4.0]]).prims[0]

        randomizer = VolumeStackRandomizer.__new__(VolumeStackRandomizer)
        randomizer._rng = np.random.default_rng(7)
        randomizer._reset_requested = False
        randomizer._physx_dt = 1.0 / 60.0
        randomizer._render_simulation = False

        with (
            patch.object(VolumeStackRandomizer, "stage", stage, create=True),
            patch.object(VolumeStackRandomizer, "prim_path", "/World/SmallSurface", create=True),
            patch("isaacsim.replicator.behavior.behaviors.volume_stack_randomizer.carb.log_warn") as warn,
        ):
            await randomizer._start_batched_asset_drop_async([(surface, asset)], drop_height=1.0, sim_steps=0)

        warn.assert_called_once()
        translation, _ = xform_utils.get_local_pose(asset, device="cpu")
        np.testing.assert_allclose(translation.numpy(), [0.0, 0.0, 1.5], atol=1e-5)

    async def test_order_range_helpers_swap_inverted_bounds(self) -> None:
        """Swap inverted vector and scalar range bounds while leaving valid ranges unchanged."""
        min_position, max_position = order_range_vectors(
            Gf.Vec3d(5.0, 0.0, 3.0),
            Gf.Vec3d(1.0, 2.0, 3.0),
            labels=("x", "y", "z"),
            owner="test",
        )
        self.assertEqual(min_position, Gf.Vec3d(1.0, 0.0, 3.0))
        self.assertEqual(max_position, Gf.Vec3d(5.0, 2.0, 3.0))

        ordered = order_scalar_range(Gf.Vec2f(45.0, 0.0), owner="test", label="textureRotateRange")
        self.assertEqual(ordered, Gf.Vec2f(0.0, 45.0))
        valid_range = Gf.Vec2f(0.1, 1.0)
        self.assertIs(order_scalar_range(valid_range, owner="test", label="textureScaleRange"), valid_range)

    async def test_csv_asset_url_helpers_skip_relative_paths_without_root(self) -> None:
        """Keep absolute CSV URLs and skip relative paths when the assets root is missing."""
        csv_entries = "file:///tmp/local.png,/Isaac/Materials/Textures/Patterns/nv_brick_grey.jpg"
        self.assertTrue(csv_has_relative_asset_url(csv_entries))
        self.assertFalse(csv_has_relative_asset_url("file:///tmp/local.png,https://example.com/a.png"))
        self.assertEqual(
            resolve_csv_asset_urls(csv_entries, None, owner="test"),
            ["file:///tmp/local.png"],
        )
        self.assertEqual(
            resolve_csv_asset_urls(csv_entries, "https://assets.example", owner="test"),
            [
                "file:///tmp/local.png",
                "https://assets.example/Isaac/Materials/Textures/Patterns/nv_brick_grey.jpg",
            ],
        )

    async def test_location_randomizer_swaps_inverted_range_before_sampling(self) -> None:
        """Order inverted location bounds in setup so sampling does not raise."""
        stage = stage_utils.get_current_stage()
        prim = stage_utils.define_prim("/World/LocationRangeTarget", "Xform")
        randomizer = LocationRandomizer.__new__(LocationRandomizer)
        randomizer._valid_prims = [prim]
        randomizer._initial_locations = {prim: Gf.Vec3d(0.0)}
        randomizer._target_offsets = {}
        randomizer._target_prim = None
        randomizer._rng = None
        exposed = {
            "range:minPosition": Gf.Vec3d(5.0, 0.0, 0.0),
            "range:maxPosition": Gf.Vec3d(1.0, 2.0, 2.0),
            "frame:useRelativeFrame": False,
            "frame:targetPrimPath": "",
            "includeChildren": False,
            "interval": 0,
            "seed": 0,
        }
        with (
            patch.object(LocationRandomizer, "prim_path", "/World/LocationRangeTarget", create=True),
            patch.object(randomizer, "_get_exposed_variable", side_effect=exposed.get),
        ):
            randomizer._setup()
        self.assertEqual(randomizer._min_position, Gf.Vec3d(1.0, 0.0, 0.0))
        self.assertEqual(randomizer._max_position, Gf.Vec3d(5.0, 2.0, 2.0))
        randomizer._rng = np.random.default_rng(42)
        with patch.object(LocationRandomizer, "stage", stage, create=True):
            randomizer._randomize_location(prim)

    async def test_rotation_randomizer_swaps_inverted_range_before_sampling(self) -> None:
        """Order inverted rotation bounds in setup so sampling does not raise."""
        stage = stage_utils.get_current_stage()
        prim = stage_utils.define_prim("/World/RotationRangeTarget", "Xform")
        randomizer = RotationRandomizer.__new__(RotationRandomizer)
        randomizer._valid_prims = [prim]
        randomizer._initial_rotations = {}
        randomizer._rng = None
        exposed = {
            "range:minRotation": Gf.Vec3d(180.0, 0.0, 0.0),
            "range:maxRotation": Gf.Vec3d(90.0, 360.0, 360.0),
            "includeChildren": False,
            "interval": 0,
            "seed": 0,
        }
        with (
            patch.object(RotationRandomizer, "prim_path", "/World/RotationRangeTarget", create=True),
            patch.object(randomizer, "_get_exposed_variable", side_effect=exposed.get),
        ):
            randomizer._setup()
        self.assertEqual(randomizer._min_rotation, Gf.Vec3d(90.0, 0.0, 0.0))
        self.assertEqual(randomizer._max_rotation, Gf.Vec3d(180.0, 360.0, 360.0))
        randomizer._rng = np.random.default_rng(42)
        with patch.object(RotationRandomizer, "stage", stage, create=True):
            randomizer._randomize_rotation(prim)

    async def test_texture_randomizer_skips_relative_urls_when_asset_root_unavailable(self) -> None:
        """Keep absolute texture URLs and skip relative CSV entries when the assets root raises."""
        stage = stage_utils.get_current_stage()
        prim = Cube("/World/TextureRootTarget").prims[0]
        randomizer = TextureRandomizer.__new__(TextureRandomizer)
        randomizer._valid_prims = []
        randomizer._rng = None
        randomizer._interval = 0
        randomizer._update_counter = 0
        randomizer._texture_urls = ["stale"]
        randomizer._texture_materials = []
        randomizer._initial_materials = {}
        exposed = {
            "includeChildren": False,
            "interval": 0,
            "textures:assets": [],
            "textures:csv": "file:///tmp/local.png,/Isaac/Materials/Textures/Patterns/nv_brick_grey.jpg",
            "projectUvwProbability": 0.9,
            "textureScaleRange": Gf.Vec2f(0.1, 1.0),
            "textureRotateRange": Gf.Vec2f(0.0, 45.0),
            "seed": 0,
        }
        with (
            patch.object(TextureRandomizer, "stage", stage, create=True),
            patch.object(TextureRandomizer, "prim", prim, create=True),
            patch.object(TextureRandomizer, "prim_path", "/World/TextureRootTarget", create=True),
            patch.object(randomizer, "_get_exposed_variable", side_effect=exposed.get),
            patch.object(randomizer, "_create_materials"),
            patch(
                "isaacsim.replicator.behavior.behaviors.texture_randomizer.get_assets_root_path",
                side_effect=RuntimeError("The '/persistent/isaac/asset_root/default' setting is not set"),
            ),
        ):
            randomizer._setup()
        self.assertEqual(randomizer._texture_urls, ["file:///tmp/local.png"])

    async def test_texture_randomizer_skips_asset_root_lookup_for_absolute_csv(self) -> None:
        """Do not resolve the assets root when every CSV texture URL is already absolute."""
        stage = stage_utils.get_current_stage()
        prim = Cube("/World/TextureAbsoluteTarget").prims[0]
        randomizer = TextureRandomizer.__new__(TextureRandomizer)
        randomizer._valid_prims = []
        randomizer._rng = None
        randomizer._interval = 0
        randomizer._update_counter = 0
        randomizer._texture_urls = []
        randomizer._texture_materials = []
        randomizer._initial_materials = {}
        exposed = {
            "includeChildren": False,
            "interval": 0,
            "textures:assets": [],
            "textures:csv": "file:///tmp/local.png,https://example.com/a.png",
            "projectUvwProbability": 0.9,
            "textureScaleRange": Gf.Vec2f(0.1, 1.0),
            "textureRotateRange": Gf.Vec2f(0.0, 45.0),
            "seed": 0,
        }
        with (
            patch.object(TextureRandomizer, "stage", stage, create=True),
            patch.object(TextureRandomizer, "prim", prim, create=True),
            patch.object(TextureRandomizer, "prim_path", "/World/TextureAbsoluteTarget", create=True),
            patch.object(randomizer, "_get_exposed_variable", side_effect=exposed.get),
            patch.object(randomizer, "_create_materials"),
            patch(
                "isaacsim.replicator.behavior.behaviors.texture_randomizer.get_assets_root_path",
            ) as mock_root,
        ):
            randomizer._setup()
        mock_root.assert_not_called()
        self.assertEqual(randomizer._texture_urls, ["file:///tmp/local.png", "https://example.com/a.png"])

    async def test_volume_stack_setup_publishes_when_asset_root_unavailable(self) -> None:
        """Publish SETUP after skipping relative CSV entries when the assets root raises."""
        stage = stage_utils.get_current_stage()
        prim = Cube("/World/VolumeRootTarget").prims[0]
        randomizer = VolumeStackRandomizer.__new__(VolumeStackRandomizer)
        randomizer._rng = None
        randomizer._state = BehaviorState.INIT
        randomizer._event_stream = None
        randomizer._event_name_out = "test.out"
        randomizer._valid_prims = []
        published: list[BehaviorState] = []

        def capture_state(new_state: BehaviorState) -> None:
            randomizer._state = new_state
            published.append(new_state)

        exposed = {
            "includeChildren": False,
            "event:output": "test.out",
            "assets:assets": [],
            "assets:csv": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd",
            "assets:numRange": Gf.Vec2i(4, 8),
            "dropHeight": 2.0,
            "renderSimulation": False,
            "removeRigidBodyDynamics": True,
            "preserveSimulationState": False,
            "seed": 0,
        }
        with (
            patch.object(VolumeStackRandomizer, "stage", stage, create=True),
            patch.object(VolumeStackRandomizer, "prim", prim, create=True),
            patch.object(VolumeStackRandomizer, "prim_path", "/World/VolumeRootTarget", create=True),
            patch.object(randomizer, "_get_exposed_variable", side_effect=exposed.get),
            patch.object(randomizer, "_set_state_and_publish", side_effect=capture_state),
            patch(
                "isaacsim.replicator.behavior.behaviors.volume_stack_randomizer.get_assets_root_path_async",
                AsyncMock(side_effect=RuntimeError("The '/persistent/isaac/asset_root/default' setting is not set")),
            ),
        ):
            await randomizer._setup_async()
        self.assertEqual(published, [BehaviorState.SETUP])
        self.assertEqual(randomizer._state, BehaviorState.SETUP)

    async def test_volume_stack_setup_skips_asset_root_lookup_for_absolute_csv(self) -> None:
        """Do not resolve the assets root when every CSV asset URL is already absolute."""
        stage = stage_utils.get_current_stage()
        prim = Cube("/World/VolumeAbsoluteTarget").prims[0]
        randomizer = VolumeStackRandomizer.__new__(VolumeStackRandomizer)
        randomizer._rng = None
        randomizer._state = BehaviorState.INIT
        randomizer._event_stream = None
        randomizer._event_name_out = "test.out"
        randomizer._valid_prims = []
        captured: dict[str, list[str]] = {}

        def capture_env(assets_urls: list[str], height: float, num_assets_range: Gf.Vec2i) -> None:
            captured["urls"] = list(assets_urls)

        exposed = {
            "includeChildren": False,
            "event:output": "test.out",
            "assets:assets": [],
            "assets:csv": "file:///tmp/box.usd",
            "assets:numRange": Gf.Vec2i(4, 8),
            "dropHeight": 2.0,
            "renderSimulation": False,
            "removeRigidBodyDynamics": True,
            "preserveSimulationState": False,
            "seed": 0,
        }
        with (
            patch.object(VolumeStackRandomizer, "stage", stage, create=True),
            patch.object(VolumeStackRandomizer, "prim", prim, create=True),
            patch.object(VolumeStackRandomizer, "prim_path", "/World/VolumeAbsoluteTarget", create=True),
            patch.object(randomizer, "_get_exposed_variable", side_effect=exposed.get),
            patch.object(randomizer, "_create_sim_environment", side_effect=capture_env),
            patch.object(randomizer, "_set_state_and_publish"),
            patch(
                "isaacsim.replicator.behavior.behaviors.volume_stack_randomizer.get_assets_root_path_async",
                new_callable=AsyncMock,
            ) as mock_root,
        ):
            await randomizer._setup_async()
        mock_root.assert_not_called()
        self.assertEqual(captured["urls"], ["file:///tmp/box.usd"])

    async def test_resolve_behavior_rng_honors_seed_change_and_injection(self) -> None:
        """Recreate the RNG when the USD seed changes, but keep an injected generator otherwise."""
        rng, last_seed, injected = resolve_behavior_rng(None, 7)
        self.assertEqual(last_seed, 7)
        self.assertFalse(injected)
        self.assertEqual(rng.integers(0, 10**9), np.random.default_rng(7).integers(0, 10**9))

        same_rng, last_seed, injected = resolve_behavior_rng(rng, 7, last_seed=7)
        self.assertIs(same_rng, rng)
        self.assertEqual(last_seed, 7)

        new_rng, last_seed, injected = resolve_behavior_rng(rng, 11, last_seed=7)
        self.assertIsNot(new_rng, rng)
        self.assertEqual(last_seed, 11)
        self.assertFalse(injected)
        self.assertEqual(new_rng.integers(0, 10**9), np.random.default_rng(11).integers(0, 10**9))

        injected_rng = np.random.default_rng(99)
        kept, last_seed, injected = resolve_behavior_rng(injected_rng, 7, last_seed=None, injected=True)
        self.assertIs(kept, injected_rng)
        self.assertTrue(injected)
        self.assertEqual(last_seed, 7)

        replaced, last_seed, injected = resolve_behavior_rng(injected_rng, 13, last_seed=7, injected=True)
        self.assertIsNot(replaced, injected_rng)
        self.assertFalse(injected)
        self.assertEqual(replaced.integers(0, 10**9), np.random.default_rng(13).integers(0, 10**9))

    async def test_location_randomizer_applies_updated_seed_on_resume(self) -> None:
        """Apply a changed USD seed on setup even when prims are already cached."""
        prim = stage_utils.define_prim("/World/LocationSeedTarget", "Xform")
        randomizer = LocationRandomizer.__new__(LocationRandomizer)
        randomizer._valid_prims = [prim]
        randomizer._initial_locations = {prim: Gf.Vec3d(0.0)}
        randomizer._target_offsets = {}
        randomizer._target_prim = None
        randomizer._rng = np.random.default_rng(1)
        randomizer._last_seed = 1
        randomizer._rng_injected = False
        exposed = {
            "range:minPosition": Gf.Vec3d(0.0, 0.0, 0.0),
            "range:maxPosition": Gf.Vec3d(1.0, 1.0, 1.0),
            "frame:useRelativeFrame": False,
            "frame:targetPrimPath": "",
            "includeChildren": False,
            "interval": 0,
            "seed": 2,
        }
        with (
            patch.object(LocationRandomizer, "prim_path", "/World/LocationSeedTarget", create=True),
            patch.object(randomizer, "_get_exposed_variable", side_effect=exposed.get),
        ):
            randomizer._setup()
        self.assertEqual(randomizer._last_seed, 2)
        self.assertFalse(randomizer._rng_injected)
        self.assertEqual(randomizer._rng.integers(0, 10**9), np.random.default_rng(2).integers(0, 10**9))

    async def test_location_randomizer_keeps_injected_rng_when_seed_unchanged(self) -> None:
        """Keep a set_rng generator through setup until the USD seed changes."""
        prim = stage_utils.define_prim("/World/LocationInjectedRng", "Xform")
        randomizer = LocationRandomizer.__new__(LocationRandomizer)
        randomizer._valid_prims = [prim]
        randomizer._initial_locations = {prim: Gf.Vec3d(0.0)}
        randomizer._target_offsets = {}
        randomizer._target_prim = None
        injected = np.random.default_rng(99)
        randomizer.set_rng(injected)
        exposed = {
            "range:minPosition": Gf.Vec3d(0.0, 0.0, 0.0),
            "range:maxPosition": Gf.Vec3d(1.0, 1.0, 1.0),
            "frame:useRelativeFrame": False,
            "frame:targetPrimPath": "",
            "includeChildren": False,
            "interval": 0,
            "seed": 1,
        }
        with (
            patch.object(LocationRandomizer, "prim_path", "/World/LocationInjectedRng", create=True),
            patch.object(randomizer, "_get_exposed_variable", side_effect=exposed.get),
        ):
            randomizer._setup()
        self.assertIs(randomizer._rng, injected)
        self.assertTrue(randomizer._rng_injected)
        self.assertEqual(randomizer._last_seed, 1)

    async def test_volume_stack_setup_reseeds_when_usd_seed_changes(self) -> None:
        """Recreate the volume-stack RNG when setup runs again with a new USD seed."""
        stage = stage_utils.get_current_stage()
        prim = Cube("/World/VolumeSeedTarget").prims[0]
        randomizer = VolumeStackRandomizer.__new__(VolumeStackRandomizer)
        randomizer._rng = np.random.default_rng(1)
        randomizer._last_seed = 1
        randomizer._rng_injected = False
        randomizer._state = BehaviorState.INIT
        randomizer._event_stream = None
        randomizer._event_name_out = "test.out"
        randomizer._valid_prims = []
        exposed = {
            "includeChildren": False,
            "event:output": "test.out",
            "assets:assets": [],
            "assets:csv": "file:///tmp/box.usd",
            "assets:numRange": Gf.Vec2i(4, 8),
            "dropHeight": 2.0,
            "renderSimulation": False,
            "removeRigidBodyDynamics": True,
            "preserveSimulationState": False,
            "seed": 2,
        }
        with (
            patch.object(VolumeStackRandomizer, "stage", stage, create=True),
            patch.object(VolumeStackRandomizer, "prim", prim, create=True),
            patch.object(VolumeStackRandomizer, "prim_path", "/World/VolumeSeedTarget", create=True),
            patch.object(randomizer, "_get_exposed_variable", side_effect=exposed.get),
            patch.object(randomizer, "_create_sim_environment"),
            patch.object(randomizer, "_set_state_and_publish"),
            patch(
                "isaacsim.replicator.behavior.behaviors.volume_stack_randomizer.get_assets_root_path_async",
                new_callable=AsyncMock,
            ),
        ):
            await randomizer._setup_async()
        self.assertEqual(randomizer._last_seed, 2)
        self.assertFalse(randomizer._rng_injected)
        self.assertEqual(randomizer._rng.integers(0, 10**9), np.random.default_rng(2).integers(0, 10**9))

    async def test_texture_randomizer_teardown_skips_when_stage_is_none(self) -> None:
        """Skip texture material restore and delete when the stage is already closed."""
        randomizer = TextureRandomizer.__new__(TextureRandomizer)
        randomizer._texture_materials = [object()]
        randomizer._initial_materials = {object(): object()}
        randomizer._valid_prims = [object()]
        randomizer._update_counter = 1
        randomizer._rng = object()
        with (
            patch.object(TextureRandomizer, "stage", None, create=True),
            patch.object(TextureRandomizer, "prim_path", "/World/ClosedStage", create=True),
        ):
            randomizer._reset()
        self.assertEqual(randomizer._texture_materials, [])
        self.assertEqual(randomizer._initial_materials, {})
        self.assertEqual(randomizer._valid_prims, [])
        self.assertIsNone(randomizer._rng)

    async def test_light_randomizer_apply_skips_prims_without_light_api(self) -> None:
        """Skip light writes on cached prims that no longer have LightAPI."""
        xform = stage_utils.define_prim("/World/NotALight", "Xform")
        light = stage_utils.define_prim("/World/IsALight", "SphereLight")
        prim_utils.create_prim_attribute(light, name="inputs:intensity", type_name=Sdf.ValueTypeNames.Float)
        prim_utils.create_prim_attribute(light, name="inputs:color", type_name=Sdf.ValueTypeNames.Color3f)
        prim_utils.set_prim_attribute_value(light, "inputs:intensity", 1.0)
        prim_utils.set_prim_attribute_value(light, "inputs:color", Gf.Vec3f(0.0, 0.0, 0.0))

        randomizer = LightRandomizer.__new__(LightRandomizer)
        randomizer._valid_prims = [xform, light]
        randomizer._rng = np.random.default_rng(0)
        randomizer._min_color = Gf.Vec3d(1.0, 1.0, 1.0)
        randomizer._max_color = Gf.Vec3d(1.0, 1.0, 1.0)
        randomizer._intensity_range = Gf.Vec2f(10.0, 10.0)

        randomizer._apply_behavior()

        self.assertEqual(prim_utils.get_prim_attribute_value(light, "inputs:intensity"), 10.0)
        np.testing.assert_allclose(list(prim_utils.get_prim_attribute_value(light, "inputs:color")), [1.0, 1.0, 1.0])
        self.assertFalse(xform.HasAttribute("inputs:intensity"))
        self.assertFalse(xform.HasAttribute("inputs:color"))
