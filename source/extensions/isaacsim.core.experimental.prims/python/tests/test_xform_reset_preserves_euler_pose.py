# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for canonicalizing XformPrim transformation operations."""

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.test
from isaacsim.core.experimental.prims import XformPrim
from pxr import Gf, Usd, UsdGeom


class TestXformResetPreservesEulerPose(omni.kit.test.AsyncTestCase):
    """Verify reset_xform_op_properties preserves Euler-authored world poses."""

    async def test_rotate_xyz_world_pose_is_preserved(self) -> None:
        """Preserve the world transform when replacing rotateXYZ with orient."""
        await stage_utils.create_new_stage_async()
        stage = stage_utils.get_current_stage(backend="usd")
        UsdGeom.Xform.Define(stage, "/World")
        prim = UsdGeom.Xform.Define(stage, "/World/TestXform").GetPrim()
        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(20.0, -35.0, 70.0))
        xformable.AddScaleOp().Set(Gf.Vec3f(0.5, 1.25, 2.0))

        before = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        XformPrim("/World/TestXform").reset_xform_op_properties()

        after = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        self.assertTrue(np.allclose(np.asarray(before), np.asarray(after), rtol=1e-6, atol=1e-6))

        property_names = prim.GetPropertyNames()
        self.assertNotIn("xformOp:rotateXYZ", property_names)
        self.assertIn("xformOp:orient", property_names)
        self.assertIn("xformOp:translate", property_names)
        self.assertIn("xformOp:scale", property_names)
