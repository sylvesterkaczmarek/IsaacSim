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

"""Tests for the IsaacReadCameraInfo node's cameraIntrinsics output."""

import numpy as np
import omni.graph.core as og
import omni.graph.core.tests as ogts
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import UsdGeom

_GRAPH_PATH = "/ActionGraph"
_NODE_NAME = "readCameraInfo"
_CAMERA_PATH = "/World/Camera"


class TestIsaacReadCameraInfo(ogts.OmniGraphTestCase):
    """Verify the cameraIntrinsics output of IsaacReadCameraInfo."""

    async def setUp(self) -> None:
        """Create a stage and acquire its timeline."""
        await omni.usd.get_context().new_stage_async()
        self._stage = omni.usd.get_context().get_stage()
        self._timeline = omni.timeline.get_timeline_interface()

    async def tearDown(self) -> None:
        """Stop the timeline after each test."""
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

    async def _read_intrinsics(
        self,
        focal_length: float,
        horiz_aperture: float,
        width: int,
        height: int,
        horiz_offset: float = 0.0,
        vert_offset: float = 0.0,
    ) -> np.ndarray:
        """Compute camera intrinsics with the IsaacReadCameraInfo node.

        Args:
            focal_length: Camera focal length.
            horiz_aperture: Camera horizontal aperture.
            width: Render product width in pixels.
            height: Render product height in pixels.
            horiz_offset: Horizontal aperture offset.
            vert_offset: Vertical aperture offset.

        Returns:
            Camera intrinsics as a three-by-three matrix.
        """
        camera = UsdGeom.Camera.Define(self._stage, _CAMERA_PATH)
        camera.GetFocalLengthAttr().Set(focal_length)
        camera.GetHorizontalApertureAttr().Set(horiz_aperture)
        camera.GetHorizontalApertureOffsetAttr().Set(horiz_offset)
        camera.GetVerticalApertureOffsetAttr().Set(vert_offset)

        render_product = rep.create.render_product(_CAMERA_PATH, (width, height))

        graph, _, _, _ = og.Controller.edit(
            {"graph_path": _GRAPH_PATH, "evaluator_name": "push"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    (_NODE_NAME, "isaacsim.core.nodes.IsaacReadCameraInfo"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    (f"{_NODE_NAME}.inputs:renderProductPath", render_product.path),
                ],
            },
        )
        self._timeline.play()
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()
        await og.Controller.evaluate(graph)

        return np.array(og.Controller.get(f"{_GRAPH_PATH}/{_NODE_NAME}.outputs:cameraIntrinsics")).reshape(3, 3)

    async def test_intrinsics_known_camera(self) -> None:
        """fx/fy/cx/cy match the closed-form values for a known camera + resolution."""
        width, height = 1280, 720
        focal_length, horiz_aperture = 50.0, 20.955
        K = await self._read_intrinsics(focal_length, horiz_aperture, width, height)

        vert_aperture = horiz_aperture * (height / width)
        expected_fx = width * focal_length / horiz_aperture
        expected_fy = height * focal_length / vert_aperture
        # The USD tenths-of-units scale cancels in the focal/aperture ratio, so raw mm values apply.
        self.assertAlmostEqual(K[0, 0], expected_fx, places=2, msg="fx mismatch")
        self.assertAlmostEqual(K[1, 1], expected_fy, places=2, msg="fy mismatch")
        self.assertAlmostEqual(K[0, 2], width * 0.5, places=2, msg="cx mismatch")
        self.assertAlmostEqual(K[1, 2], height * 0.5, places=2, msg="cy mismatch")
        self.assertAlmostEqual(K[0, 1], 0.0, places=6, msg="K[0][1] should be 0")
        self.assertAlmostEqual(K[2, 2], 1.0, places=6, msg="K[2][2] should be 1")

    async def test_intrinsics_with_aperture_offset(self) -> None:
        """A non-zero aperture offset shifts the principal point (cx/cy) off center."""
        width, height = 1280, 720
        focal_length, horiz_aperture = 50.0, 20.955
        horiz_offset, vert_offset = 2.0, -1.5
        K = await self._read_intrinsics(
            focal_length, horiz_aperture, width, height, horiz_offset=horiz_offset, vert_offset=vert_offset
        )

        vert_aperture = horiz_aperture * (height / width)
        # offset/aperture is the fractional shift across the sensor, scaled to pixels.
        expected_cx = width * (0.5 + horiz_offset / horiz_aperture)
        expected_cy = height * (0.5 + vert_offset / vert_aperture)
        self.assertAlmostEqual(K[0, 2], expected_cx, places=2, msg="cx should include horizontal offset")
        self.assertAlmostEqual(K[1, 2], expected_cy, places=2, msg="cy should include vertical offset")
        # fx/fy are unaffected by the offset.
        self.assertAlmostEqual(K[0, 0], width * focal_length / horiz_aperture, places=2, msg="fx mismatch")
        self.assertAlmostEqual(K[1, 1], height * focal_length / vert_aperture, places=2, msg="fy mismatch")

    async def test_negative_aperture_returns_identity(self) -> None:
        """A non-positive horizontal aperture yields an identity intrinsics matrix (guard)."""
        K = await self._read_intrinsics(focal_length=50.0, horiz_aperture=-1.0, width=1280, height=720)
        np.testing.assert_allclose(K, np.eye(3), atol=1e-6, err_msg="expected identity for non-positive aperture")
