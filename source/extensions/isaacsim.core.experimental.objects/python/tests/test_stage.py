# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Validate the ``Stage`` class lifecycle, prim operations, and stage metadata.

The suite exercises stage creation and teardown (including template support),
prim define/move/remove, reference loading, up-axis/units/time-code getters and
setters, and string-representation generation.
"""

import os
import tempfile

import numpy as np
import omni.kit.test
from isaacsim.core.experimental.objects import Stage
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdUtils


class TestStage(omni.kit.test.AsyncTestCase):
    """Exercise Stage instance dispatch."""

    async def setUp(self) -> None:
        """Method called to prepare the test fixture."""
        super().setUp()

    async def tearDown(self) -> None:
        """Method called immediately after the test method has been called."""
        # ------------------
        Stage().close_stage()
        # ------------------
        super().tearDown()

    async def test_open_stage(self) -> None:
        """Test open stage."""
        assets_root_path = await get_assets_root_path_async(skip_check=True)
        # test cases
        # - sync
        stage = Stage().open_stage(
            usd_path=assets_root_path + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda",
        )
        self.assertTrue(stage.is_valid())
        self.assertTrue(omni.usd.get_context().get_stage().GetPrimAtPath("/panda/panda_hand").IsValid())
        # - async
        stage = await Stage().create_stage_async()
        opened_stage = await stage.open_stage_async(
            usd_path=assets_root_path + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda",
        )
        self.assertTrue(opened_stage.is_valid())
        self.assertTrue(omni.usd.get_context().get_stage().GetPrimAtPath("/panda/panda_hand").IsValid())

    async def test_save_close_stage(self) -> None:
        """Test save close stage."""
        assets_root_path = await get_assets_root_path_async(skip_check=True)
        # create and populate stage
        stage = await Stage().create_stage_async()
        self.assertIsNotNone(omni.usd.get_context().get_stage())
        stage.add_reference(
            usd_path=assets_root_path + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda",
            path="/World/panda",
            variants={"Gripper": "alternatefinger", "Mesh": "performance"},
        )
        # save and close stage, then open it again
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_file = os.path.join(tmp_dir, "test_save_close_stage.usd")
            # - save stage
            stage.save_stage(usd_path=tmp_file)
            self.assertTrue(os.path.exists(tmp_file) and os.path.isfile(tmp_file))
            # - close stage
            self.assertTrue(stage.close_stage())
            self.assertFalse(stage.is_valid() or omni.usd.get_context().get_stage() is not None)
            # - open stage
            opened_stage = stage.open_stage(usd_path=tmp_file)
            opened_usd_stage = omni.usd.get_context().get_stage()
            self.assertTrue(opened_stage.is_valid())
            self.assertIsNotNone(opened_usd_stage)
            self.assertTrue(opened_usd_stage.GetPrimAtPath("/World/panda/panda_hand").IsValid())

    async def test_add_reference_to_stage(self) -> None:
        """Test add reference to stage."""
        assets_root_path = await get_assets_root_path_async(skip_check=True)
        # create and populate stage
        stage = await Stage().create_stage_async()
        stage.add_reference(
            usd_path=assets_root_path + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda",
            path="/World/panda",
            variants={"Gripper": "alternatefinger", "Mesh": "performance"},
        )
        usd_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/panda")
        self.assertIsInstance(usd_prim, Usd.Prim)
        self.assertEqual(usd_prim.GetPath(), "/World/panda")
        self.assertEqual(usd_prim.GetVariantSet("Gripper").GetVariantSelection(), "alternatefinger")
        self.assertEqual(usd_prim.GetVariantSet("Mesh").GetVariantSelection(), "performance")

    async def test_define_prim(self) -> None:
        """Test define prim."""
        stage = await Stage().create_stage_async()
        specs = [
            # UsdGeomTokensType
            ("Camera", UsdGeom.Camera),
            ("Capsule", UsdGeom.Capsule),
            ("Cone", UsdGeom.Cone),
            ("Cube", UsdGeom.Cube),
            ("Cylinder", UsdGeom.Cylinder),
            ("Mesh", UsdGeom.Mesh),
            ("Plane", UsdGeom.Plane),
            ("Points", UsdGeom.Points),
            ("Scope", UsdGeom.Scope),
            ("Sphere", UsdGeom.Sphere),
            ("Xform", UsdGeom.Xform),
            # UsdLuxTokensType
            ("CylinderLight", UsdLux.CylinderLight),
            ("DiskLight", UsdLux.DiskLight),
            ("DistantLight", UsdLux.DistantLight),
            ("DomeLight", UsdLux.DomeLight),
            ("RectLight", UsdLux.RectLight),
            ("SphereLight", UsdLux.SphereLight),
            # UsdPhysicsTokensType
            ("PhysicsScene", UsdPhysics.Scene),
        ]
        # USD prim
        for token, prim_type in specs:
            path = stage.define_prim(f"/{token}", type_name=token)
            usd_prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
            self.assertTrue(usd_prim.IsA(prim_type), f"Prim ({path}) is not a {prim_type}")
        # exceptions
        # - non-absolute path
        self.assertRaises(ValueError, stage.define_prim, "World")
        # - non-valid path
        self.assertRaises(ValueError, stage.define_prim, "/World/")
        # - prim already exists with a different type
        self.assertRaises(RuntimeError, stage.define_prim, "/Sphere", type_name="Cube")

    async def test_remove_prim(self) -> None:
        """Test remove prim."""
        assets_root_path = await get_assets_root_path_async(skip_check=True)
        # create and populate stage
        stage = await Stage().create_stage_async()
        prim = stage.define_prim("/World/A", "Xform")
        stage.add_reference(
            usd_path=assets_root_path + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda",
            path="/World/panda",
            variants={"Gripper": "alternatefinger", "Mesh": "performance"},
        )
        # test cases
        self.assertTrue(stage.remove_prim(prim))
        self.assertFalse(stage.remove_prim("/World/panda/panda_hand"))  # ancestral prim (cannot be removed)
        self.assertTrue(stage.remove_prim("/World/panda"))
        # exceptions
        self.assertRaisesRegex(ValueError, "not a valid prim", stage.remove_prim, "/World/A")

    async def test_move_prim(self) -> None:
        """Test move prim."""
        stage = stage = await Stage().create_stage_async()
        prim_a = stage.define_prim("/World/A", "Xform")
        prim_b = stage.define_prim("/World/B", "Xform")
        # test cases
        usd_stage = omni.usd.get_context().get_stage()
        # - move A to (inside) B
        result, path = stage.move_prim(prim_a, prim_b)
        self.assertTrue(result)
        self.assertEqual(path, "/World/B/A")
        self.assertEqual(usd_stage.GetPrimAtPath("/World/B/A").IsValid(), True)
        self.assertEqual(usd_stage.GetPrimAtPath("/World/A").IsValid(), False)
        # - move A next to B
        result, path = stage.move_prim("/World/B/A", "/World")
        self.assertTrue(result)
        self.assertEqual(path, "/World/A")
        self.assertEqual(usd_stage.GetPrimAtPath("/World/A").IsValid(), True)
        self.assertEqual(usd_stage.GetPrimAtPath("/World/B/A").IsValid(), False)
        # - move A to root
        result, path = stage.move_prim("/World/A", "/")
        self.assertTrue(result)
        self.assertEqual(path, "/A")
        self.assertEqual(usd_stage.GetPrimAtPath("/A").IsValid(), True)
        self.assertEqual(usd_stage.GetPrimAtPath("/World/A").IsValid(), False)
        # - move A to an unexisting path
        result, path = stage.move_prim("/A", "/World/C")
        self.assertTrue(result)
        self.assertEqual(path, "/World/C")
        self.assertEqual(usd_stage.GetPrimAtPath("/World/C").IsValid(), True)
        self.assertEqual(usd_stage.GetPrimAtPath("/A").IsValid(), False)
        # exceptions
        # - prim is not a valid prim
        self.assertRaisesRegex(ValueError, "not a valid prim", stage.move_prim, "/ABC", "/")
        # - destination path is not a valid path string
        self.assertRaisesRegex(ValueError, "not a valid path string", stage.move_prim, "/World/B", "?")
        # - destination path has unexisting parents
        self.assertRaisesRegex(ValueError, "unexisting parent", stage.move_prim, "/World/B", "/World/X/Y")

    async def test_stage_units(self) -> None:
        """Test stage units."""
        stage = await Stage().create_stage_async()
        # test cases
        # - default units
        self.assertEqual(stage.get_units(), (1.0, 1.0))
        # - random units
        for meters_per_unit, kilograms_per_unit in np.random.rand(10, 2):
            stage.set_units(meters_per_unit=meters_per_unit, kilograms_per_unit=kilograms_per_unit)
            self.assertEqual(stage.get_units(), (meters_per_unit, kilograms_per_unit))

    async def test_stage_up_axis(self) -> None:
        """Test stage up axis."""
        stage = await Stage().create_stage_async()
        # test cases
        # - default up axis
        self.assertEqual(stage.get_up_axis(), "Z")
        # - supported up axis
        for up_axis in ["Y", "Z", "y", "z"]:
            stage.set_up_axis(up_axis)
            self.assertEqual(stage.get_up_axis(), up_axis.upper())

    async def test_stage_time_code(self) -> None:
        """Test stage time code."""
        stage = await Stage().create_stage_async()
        # test cases
        # - default time code
        self.assertEqual(stage.get_time_code(), (0.0, 100.0, 60.0))
        # - random time code
        for start_time_code, end_time_code, time_codes_per_second in np.random.rand(10, 3):
            stage.set_time_code(
                start_time_code=start_time_code,
                end_time_code=end_time_code,
                time_codes_per_second=time_codes_per_second,
            )
            self.assertEqual(stage.get_time_code(), (start_time_code, end_time_code, time_codes_per_second))
