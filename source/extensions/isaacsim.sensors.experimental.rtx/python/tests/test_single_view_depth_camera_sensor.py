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

"""Tests for the SingleViewDepthCameraSensor class."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.app
import omni.kit.test
import warp as wp
from isaacsim.core.experimental.prims.tests.common import check_allclose, cprint, draw_sample
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.sensors.experimental.rtx import (
    CameraSensor,
    RtxCamera,
    SingleViewDepthCameraSensor,
    draw_annotator_data_to_image,
)
from pxr import Gf, UsdRender

from .common import normalize_semantics
from .test_camera_sensor import parametrize, populate_stage

RESOLUTION = (256, 320)  # following OpenCV/NumPy convention (height, width)
EXPECTED_ANNOTATOR_SPEC = {
    "bounding_box_2d_loose": {"type": np.ndarray},
    "bounding_box_2d_tight": {"type": np.ndarray},
    "bounding_box_3d": {"type": np.ndarray},
    "distance_to_camera": {"channels": 1, "dtype": wp.float32, "type": wp.array},
    "distance_to_image_plane": {"channels": 1, "dtype": wp.float32, "type": wp.array},
    "instance_id_segmentation": {"channels": 1, "dtype": wp.uint32, "type": wp.array},
    "instance_segmentation": {"channels": 1, "dtype": wp.uint32, "type": wp.array},
    "motion_vectors": {"channels": 2, "dtype": wp.float32, "type": wp.array},
    "normals": {"channels": 3, "dtype": wp.float32, "type": wp.array},
    "pointcloud": {"type": wp.array},
    "semantic_segmentation": {"channels": 1, "dtype": wp.uint32, "type": wp.array},
    "semantic_segmentation": {"channels": 1, "dtype": wp.uint32, "type": wp.array},
    # single view depth sensor annotators
    "depth_sensor_distance": {"channels": 1, "dtype": wp.float32, "type": wp.array},
    "depth_sensor_imager": {"channels": 1, "dtype": wp.float32, "type": wp.array},
    "depth_sensor_point_cloud_color": {"channels": 4, "dtype": wp.uint8, "type": wp.array},
    "depth_sensor_point_cloud_position": {"channels": 3, "dtype": wp.float32, "type": wp.array},
}


class TestSingleViewDepthCameraSensor(omni.kit.test.AsyncTestCase):
    """Test case class for the SingleViewDepthCameraSensor class."""

    async def setUp(self) -> None:
        """Method called to prepare the test fixture."""
        super().setUp()
        self.maxDiff = None  # show all diffs
        self.save_images = False  # whether to save images

    async def tearDown(self) -> None:
        """Method called immediately after the test method has been called."""
        super().tearDown()

    # --------------------------------------------------------------------

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_sensor_baseline(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the sensor baseline can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for v0, expected_v0 in draw_sample(shape=(num_prims, 1), dtype=wp.float32, types=[list]):
            prim.set_sensor_baseline(np.array(v0).item())
            output = prim.get_sensor_baseline()
            self.assertAlmostEqual(expected_v0.item(), output, msg=f"Given: {v0}")

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_sensor_disparity_confidence(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the sensor disparity confidence can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for v0, expected_v0 in draw_sample(shape=(num_prims, 1), dtype=wp.float32, types=[list]):
            prim.set_sensor_disparity_confidence(np.array(v0).item())
            output = prim.get_sensor_disparity_confidence()
            self.assertAlmostEqual(expected_v0.item(), output, msg=f"Given: {v0}")

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_sensor_maximum_disparity(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the sensor maximum disparity can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for v0, expected_v0 in draw_sample(shape=(num_prims, 1), dtype=wp.float32, types=[list]):
            prim.set_sensor_maximum_disparity(np.array(v0).item())
            output = prim.get_sensor_maximum_disparity()
            self.assertAlmostEqual(expected_v0.item(), output, msg=f"Given: {v0}")

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_enabled_post_processing(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the enabled post processing flag can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for item in [False, True]:
            prim.set_enabled_post_processing(item)
            output = prim.get_enabled_post_processing()
            self.assertEqual(item, output, msg=f"Given: {item}")

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_sensor_focal_length(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the sensor focal length can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for v0, expected_v0 in draw_sample(shape=(num_prims, 1), dtype=wp.float32, types=[list]):
            prim.set_sensor_focal_length(np.array(v0).item())
            output = prim.get_sensor_focal_length()
            self.assertAlmostEqual(expected_v0.item(), output, msg=f"Given: {v0}")

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_sensor_distance_cutoffs(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the sensor distance cutoffs can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for (v0, expected_v0), (v1, expected_v1) in zip(
            draw_sample(shape=(num_prims, 1), dtype=wp.float32, types=[list]),
            draw_sample(shape=(num_prims, 1), dtype=wp.float32, types=[list]),
        ):
            prim.set_sensor_distance_cutoffs(np.array(v0).item(), np.array(v1).item())
            output = prim.get_sensor_distance_cutoffs()
            self.assertAlmostEqual(expected_v0.item(), output[0], msg=f"Given: {v0}")
            self.assertAlmostEqual(expected_v1.item(), output[1], msg=f"Given: {v1}")

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_sensor_disparity_noise_downscale(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the sensor disparity noise downscale can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for v0, expected_v0 in draw_sample(shape=(num_prims, 1), dtype=wp.float32, types=[list]):
            prim.set_sensor_disparity_noise_downscale(np.array(v0).item())
            output = prim.get_sensor_disparity_noise_downscale()
            self.assertAlmostEqual(expected_v0.item(), output, msg=f"Given: {v0}")

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_sensor_noise_parameters(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the sensor noise parameters can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for (v0, expected_v0), (v1, expected_v1) in zip(
            draw_sample(shape=(num_prims, 1), dtype=wp.float32, types=[list]),
            draw_sample(shape=(num_prims, 1), dtype=wp.float32, types=[list]),
        ):
            prim.set_sensor_noise_parameters(np.array(v0).item(), np.array(v1).item())
            output = prim.get_sensor_noise_parameters()
            self.assertAlmostEqual(expected_v0.item(), output[0], msg=f"Given: {v0}")
            self.assertAlmostEqual(expected_v1.item(), output[1], msg=f"Given: {v1}")

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_enabled_outlier_removal(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the enabled outlier removal flag can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for v0, expected_v0 in draw_sample(shape=(num_prims, 1), dtype=wp.bool, types=[list]):
            prim.set_enabled_outlier_removal(np.array(v0).item())
            output = prim.get_enabled_outlier_removal()
            self.assertAlmostEqual(expected_v0.item(), output, msg=f"Given: {v0}")

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_sensor_output_mode(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the sensor output mode can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for v0, expected_v0 in draw_sample(shape=(num_prims, 1), dtype=wp.int32, low=0, high=8, types=[list]):
            prim.set_sensor_output_mode(np.array(v0).item())
            output = prim.get_sensor_output_mode()
            self.assertEqual(expected_v0.item(), output, msg=f"Given: {v0}")

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": []},
        populate_stage_func=populate_stage,
    )
    async def test_sensor_size(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that the sensor size can be set and retrieved correctly.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        for v0, expected_v0 in draw_sample(shape=(num_prims, 1), dtype=wp.float32, types=[list]):
            prim.set_sensor_size(np.array(v0).item())
            output = prim.get_sensor_size()
            self.assertAlmostEqual(expected_v0.item(), output, msg=f"Given: {v0}")

    # --------------------------------------------------------------------

    @parametrize(
        prim_class=SingleViewDepthCameraSensor,
        prim_class_kwargs={"resolution": RESOLUTION, "annotators": list(EXPECTED_ANNOTATOR_SPEC.keys())},
        populate_stage_func=populate_stage,
    )
    async def test_data(self, prim: Any, num_prims: int, operation: str) -> None:
        """Test that sensor data is correctly retrieved for all annotators.

        Args:
            prim: Single-view depth camera sensor under test.
            num_prims: Number of camera prims created by the parametrized fixture.
            operation: Parametrized fixture operation name.
        """
        prim.camera.set_focal_lengths(1.814756)
        prim.camera.set_focus_distances(400.0)
        prim.set_sensor_baseline(55)
        prim.set_sensor_focal_length(891.0)
        prim.set_sensor_size(1280.0)
        prim.set_sensor_maximum_disparity(110.0)
        prim.set_sensor_disparity_confidence(0.99)
        prim.set_sensor_noise_parameters(0.5, 1.0)
        prim.set_sensor_disparity_noise_downscale(1.0)
        prim.set_sensor_distance_cutoffs(0.5, 9999.9)
        prim.set_enabled_post_processing(True)
        for path in prim.camera.paths:
            ViewportManager.set_camera_view(path, eye=[1.0, 0.5, 1.0], target=[0.0, 0.0, 0.25])
        # test cases
        for annotator in sorted(EXPECTED_ANNOTATOR_SPEC.keys()):
            cprint(f"  |    |-- annotator: {annotator}")
            spec = EXPECTED_ANNOTATOR_SPEC[annotator]
            data, info = None, {}
            for i in range(10):
                await app_utils.update_app_async()
                data, info = prim.get_data(annotator)
                if data is not None:
                    break
            if data is None:
                raise RuntimeError(f"No data available from '{annotator}' annotator after {i + 1} steps")
            else:
                cprint(f"  |    |    |-- data available after {i + 1} steps")

            # check data
            if annotator in [
                "bounding_box_2d_tight",
                "bounding_box_2d_loose",
                "bounding_box_3d",
                "pointcloud",
            ]:
                # - type
                self.assertIsInstance(
                    data, spec["type"], f"'{annotator}' annotator type {type(data)} != {spec['type']}"
                )
            else:
                # - shape
                shape = (*RESOLUTION, spec["channels"])
                self.assertEqual(data.shape, shape, f"'{annotator}' annotator shape {data.shape} != {shape}")
                # - type
                self.assertIsInstance(
                    data, spec["type"], f"'{annotator}' annotator type {type(data)} != {spec['type']}"
                )
                # - dtype
                dtype = spec["dtype"]
                self.assertEqual(data.dtype, dtype, f"'{annotator}' annotator dtype {data.dtype} != {dtype}")
                # - out
                out = wp.empty(shape, dtype=dtype, device=data.device)
                prim.get_data(annotator, out=out)
                check_allclose(data, out)
                # - info
                if annotator == "instance_id_segmentation":
                    cprint(f"  |    |    |-- {info}")
                    self.assertIn("idToLabels", info)
                    id_to_labels = info["idToLabels"]
                    self.assertTrue(
                        set(id_to_labels.values()).issubset(
                            {"INVALID", "/World/Cone", "/World/Cube", "/World/GroundPlane/geom", "/World/Sphere"}
                        ),
                        msg=f"Annotator info mismatch for '{annotator}' ({set(id_to_labels.values())})",
                    )
                elif annotator == "instance_segmentation":
                    cprint(f"  |    |    |-- {info}")
                    self.assertIn("idToLabels", info)
                    self.assertIn("idToSemantics", info)
                    self.assertEqual(set(info["idToLabels"].keys()), {0, 1, 2, 3, 4})
                    self.assertEqual(set(info["idToSemantics"].keys()), {0, 1, 2, 3, 4})
                    idToLabels = info["idToLabels"]
                    idToSemantics = info["idToSemantics"]
                    expected_label_to_semantics = {
                        "BACKGROUND": {"class": "BACKGROUND"},
                        "UNLABELLED": {"class": "UNLABELLED"},
                        "/World/Cone": {"shape": "cone"},
                        "/World/Cube": {"shape": "cube"},
                        "/World/Sphere": {"shape": "sphere", "class": "label_a,label_b"},
                    }
                    for expected_label, expected_semantics in expected_label_to_semantics.items():
                        id = None
                        for key, label in idToLabels.items():
                            if label == expected_label:
                                id = key
                                break
                        self.assertIsNotNone(id, f"Label '{expected_label}' not found in idToLabels")
                        self.assertEqual(
                            normalize_semantics(idToSemantics[id]), normalize_semantics(expected_semantics)
                        )
                elif annotator == "semantic_segmentation":
                    expected_values = [
                        {"class": "BACKGROUND"},
                        {"class": "UNLABELLED"},
                        {"shape": "cube"},
                        {"shape": "sphere", "class": "label_a,label_b"},
                        {"shape": "cone"},
                    ]
                    cprint(f"  |    |    |-- {info}")
                    self.assertEqual(
                        sorted(info.keys()), ["idToLabels"], msg=f"Annotator info mismatch for '{annotator}'"
                    )
                    self.assertEqual(
                        sorted(info["idToLabels"].keys()),
                        ["0", "1", "2", "3", "4"],
                        msg=f"Annotator info mismatch for '{annotator}'",
                    )
                    normalized_expected_values = [normalize_semantics(value) for value in expected_values]
                    for value in info["idToLabels"].values():
                        self.assertIn(
                            normalize_semantics(value),
                            normalized_expected_values,
                            msg=f"Annotator info mismatch for '{annotator}'",
                        )
                else:
                    self.assertDictEqual({}, info, msg=f"Annotator info mismatch for '{annotator}'")

            # render data and save it in the sub-folder `data` in the same folder as the test file
            image = draw_annotator_data_to_image(annotator=annotator, data=data, info=info)
            if self.save_images:
                filedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
                filename = f"single_view_depth_camera_sensor_{operation}_{num_prims}_{annotator}.png"
                filepath = os.path.join(filedir, filename)
                os.makedirs(filedir, exist_ok=True)
                print(f"Saving image to {filepath}")
                cv2.imwrite(filepath, image)


# render var source names authored on the fixture asset's render product, in orderedVars order
_ASSET_RENDER_VARS = [
    "DepthSensorDistance",
    "DepthSensorImager",
    "DepthSensorPointCloudColor",
    "DepthSensorPointCloudPosition",
]
# annotator keys those render vars map back to
_ASSET_ANNOTATORS = [
    "depth_sensor_distance",
    "depth_sensor_imager",
    "depth_sensor_point_cloud_color",
    "depth_sensor_point_cloud_position",
]


class TestSingleViewDepthCameraSensorAttach(omni.kit.test.AsyncTestCase):
    """Tests for attaching to a pre-authored render product in an all-inclusive USD asset (ISIM-5035)."""

    async def setUp(self) -> None:
        """Method called to prepare the test fixture."""
        super().setUp()
        self.maxDiff = None
        self._temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self._asset_baseline = 42.0
        self._asset_resolution = (480, 320)  # (height, width)
        self._asset_path = await self._build_depth_asset()

    async def tearDown(self) -> None:
        """Method called immediately after the test method has been called."""
        try:
            stage_utils.close_stage()
        except Exception:
            pass
        self._temp_dir.cleanup()
        super().tearDown()

    async def _build_depth_asset(self) -> str:
        """Author a minimal all-inclusive depth-sensor USD asset and export it to a temp file.

        The asset contains a Camera (with OmniSensorAPI), a RenderProduct with
        ``OmniSensorDepthSensorSingleViewAPI`` applied (authored resolution, camera relationship, and
        depth-sensor attributes), and the four depth-sensor RenderVar prims wired via ``orderedVars``.

        Returns:
            Path to the exported depth-sensor asset.
        """
        await stage_utils.create_new_stage_async()
        stage = stage_utils.get_current_stage()
        # asset root prim (default prim, so it composes under the reference target path)
        root = stage.DefinePrim("/Sensor", "Xform")
        stage.SetDefaultPrim(root)
        # depth camera
        camera = stage.DefinePrim("/Sensor/Depth", "Camera")
        camera.ApplyAPI("OmniSensorAPI")
        # render product with the depth-sensor schema + a depth-sensor attribute authored
        render_product = SingleViewDepthCameraSensor.add_template_render_product(
            parent_prim_path="/Sensor/TemplateRenderProduct",
            camera_prim_path="/Sensor/Depth",
            **{"omni:rtx:post:depthSensor:baselineMM": self._asset_baseline},
        )
        # authored resolution is stored as (width, height)
        UsdRender.Product(render_product).CreateResolutionAttr(
            Gf.Vec2i(self._asset_resolution[1], self._asset_resolution[0])
        )
        # render var prims declared via orderedVars
        var_paths = []
        for source_name in _ASSET_RENDER_VARS:
            var = UsdRender.Var.Define(stage, f"/Sensor/TemplateRenderProduct/{source_name}")
            var.CreateSourceNameAttr(source_name)
            var_paths.append(var.GetPath())
        UsdRender.Product(render_product).CreateOrderedVarsRel().SetTargets(var_paths)
        # export the authored asset (composite asset → .usd so RtxCamera.create references it as an Xform)
        asset_path = str(Path(self._temp_dir.name) / "depth_asset.usd")
        stage.Export(asset_path)
        return asset_path

    async def test_attaches_to_pre_authored_render_product(self) -> None:
        """Attaching to the asset's render product uses that prim directly and derives its config."""
        await stage_utils.create_new_stage_async()
        await ViewportManager.wait_for_viewport_async()
        stage_utils.define_prim("/World", "Xform")
        camera = RtxCamera.create(path="/World/Sensor", usd_path=self._asset_path)
        sensor = SingleViewDepthCameraSensor(camera)
        try:
            render_product_path = str(sensor.render_product.GetPath())
            # the sensor renders through the asset's render product, not a fresh /Render product
            self.assertEqual(render_product_path, "/World/Sensor/TemplateRenderProduct/Depth_render_product")
            self.assertFalse(render_product_path.startswith("/Render"))
            # resolution and annotators are derived from the asset
            self.assertEqual(sensor.resolution, self._asset_resolution)
            self.assertEqual(sorted(sensor.annotators), sorted(_ASSET_ANNOTATORS))
            # depth-sensor parameters are read straight from the asset's render product (no copy step)
            self.assertAlmostEqual(sensor.get_sensor_baseline(), self._asset_baseline)
        finally:
            # tearing the sensor down must not delete the asset-owned render product prim
            sensor._invalidate_sensor()
            del sensor
            await omni.kit.app.get_app().next_update_async()
        stage = stage_utils.get_current_stage()
        self.assertTrue(stage.GetPrimAtPath("/World/Sensor/TemplateRenderProduct/Depth_render_product").IsValid())

    @unittest.expectedFailure  # NVBUG-6641439
    async def test_rename_asset_preserves_attached_synthetic_data_graph(self) -> None:
        """Renaming an asset with an attached nested render product keeps its graph valid."""
        from omni.syntheticdata import SyntheticData, SyntheticDataStage

        await stage_utils.create_new_stage_async()
        await ViewportManager.wait_for_viewport_async()
        stage_utils.define_prim("/World", "Xform")
        camera = RtxCamera.create(path="/World/Sensor", usd_path=self._asset_path)
        sensor = SingleViewDepthCameraSensor(camera)
        synthetic_data = SyntheticData.Get()
        try:
            render_product_path = str(sensor.render_product.GetPath())
            node_graph = synthetic_data.get_graph(SyntheticDataStage.POST_RENDER, render_product_path)
            node_graph.get_wrapped_graph()
            moved, moved_path = stage_utils.move_prim("/World/Sensor", "/World/RenamedSensor")
            self.assertTrue(moved)
            self.assertEqual(moved_path, "/World/RenamedSensor")
            self.assertTrue(
                stage_utils.get_current_stage()
                .GetPrimAtPath("/World/RenamedSensor/TemplateRenderProduct/Depth_render_product")
                .IsValid()
            )
            node_graph.get_wrapped_graph()
        finally:
            synthetic_data.reset(usd=False)
            sensor._invalidate_sensor()

    async def test_attaches_when_camera_wrapped_directly(self) -> None:
        """A camera wrapped directly (no RtxCamera.create) still attaches via a stage-wide search."""
        await stage_utils.create_new_stage_async()
        await ViewportManager.wait_for_viewport_async()
        stage_utils.define_prim("/World", "Xform")
        # reference the asset as an Xform and wrap its depth camera directly (no _asset_root_path)
        stage_utils.add_reference_to_stage(usd_path=self._asset_path, path="/World/Sensor", prim_type="Xform")
        camera = RtxCamera("/World/Sensor/Depth")
        sensor = SingleViewDepthCameraSensor(camera)
        try:
            render_product_path = str(sensor.render_product.GetPath())
            self.assertEqual(render_product_path, "/World/Sensor/TemplateRenderProduct/Depth_render_product")
            self.assertEqual(sensor.resolution, self._asset_resolution)
            self.assertEqual(sorted(sensor.annotators), sorted(_ASSET_ANNOTATORS))
            self.assertAlmostEqual(sensor.get_sensor_baseline(), self._asset_baseline)
        finally:
            sensor._invalidate_sensor()
            del sensor
            await omni.kit.app.get_app().next_update_async()

    async def test_creates_new_render_product_without_asset(self) -> None:
        """A camera with no matching pre-authored render product falls back to creating a new one."""
        await stage_utils.create_new_stage_async()
        await ViewportManager.wait_for_viewport_async()
        stage_utils.define_prim("/World", "Xform")
        # a bare camera with no authored depth render product on the stage
        camera = RtxCamera("/World/Camera")
        sensor = SingleViewDepthCameraSensor(camera, resolution=(256, 320), annotators=["depth_sensor_distance"])
        try:
            # the create-new path produces a render product under /Render
            self.assertTrue(str(sensor.render_product.GetPath()).startswith("/Render"))
            self.assertEqual(sensor.resolution, (256, 320))
        finally:
            sensor._invalidate_sensor()
            del sensor
            await omni.kit.app.get_app().next_update_async()

    async def test_explicit_resolution_and_annotators_when_attaching(self) -> None:
        """Explicit annotators are honored, and a mismatched resolution is overridden by the asset's."""
        await stage_utils.create_new_stage_async()
        await ViewportManager.wait_for_viewport_async()
        stage_utils.define_prim("/World", "Xform")
        camera = RtxCamera.create(path="/World/Sensor", usd_path=self._asset_path)
        sensor = SingleViewDepthCameraSensor(camera, resolution=(123, 456), annotators=["depth_sensor_distance"])
        try:
            # the asset's authored resolution wins over the mismatched requested resolution
            self.assertEqual(sensor.resolution, self._asset_resolution)
            # explicitly requested annotators are used as-is (not derived from the asset)
            self.assertEqual(sensor.annotators, ["depth_sensor_distance"])
        finally:
            sensor._invalidate_sensor()
            del sensor
            await omni.kit.app.get_app().next_update_async()

    async def test_plain_camera_sensor_attaches_to_pre_authored_render_product(self) -> None:
        """A plain CameraSensor attaches to any pre-authored RenderProduct targeting its camera.

        With ``_ASSET_RP_SCHEMA`` unset, the schema is not used as a filter, so a base
        :class:`CameraSensor` still attaches to the asset's pre-authored render product
        (deriving its resolution) rather than creating a fresh one.
        """
        await stage_utils.create_new_stage_async()
        await ViewportManager.wait_for_viewport_async()
        stage_utils.define_prim("/World", "Xform")
        camera = RtxCamera.create(path="/World/Sensor", usd_path=self._asset_path)
        sensor = CameraSensor(camera, resolution=(256, 320), annotators=["rgb"])
        try:
            # no schema filter: attaches to the asset's render product instead of creating a new one
            render_product_path = str(sensor.render_product.GetPath())
            self.assertEqual(render_product_path, "/World/Sensor/TemplateRenderProduct/Depth_render_product")
            self.assertFalse(render_product_path.startswith("/Render"))
            # resolution is derived from the asset, overriding the requested value
            self.assertEqual(sensor.resolution, self._asset_resolution)
        finally:
            sensor._invalidate_sensor()
            del sensor
            await omni.kit.app.get_app().next_update_async()
