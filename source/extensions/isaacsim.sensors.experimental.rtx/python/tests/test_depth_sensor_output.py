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

"""Verify golden depth output for every supported depth-sensor camera asset."""

import gc
import re
import tempfile
from pathlib import Path
from typing import Any

import carb
import cv2
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.test
from isaacsim.core.experimental.objects import Cone, Cube, GroundPlane, Sphere, SphereLight
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.sensors.experimental.rtx import (
    SUPPORTED_CAMERA_CONFIGS,
    RtxCamera,
    SingleViewDepthCameraSensor,
    draw_annotator_data_to_image,
    get_camera_metadata,
)
from isaacsim.storage.native import get_assets_root_path
from isaacsim.test.utils.image_comparison import compare_arrays_within_tolerances
from pxr import Gf, Usd, UsdGeom

SAVE_DEBUG_IMAGES = False
UPDATE_GOLDEN_IMAGES = False

_GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "golden" / "depth_sensor_output"
_DEBUG_DIR = Path(tempfile.gettempdir()) / "isaacsim.sensors.experimental.rtx" / "depth_sensor_output"
# Depth-sensor disparity noise is stochastic. This matches the tolerance used by
# existing noisy-depth golden tests while retaining the raw-output validity checks below.
_MEAN_TOLERANCE = 30.0
_MINIMUM_UPDATES = 10
_MAXIMUM_UPDATES = 60
_FALLBACK_DEPTH_RESOLUTION = (256, 320)
_CAMERA_EYE = Gf.Vec3d(1.0, 0.5, 1.0)
_CAMERA_TARGET = Gf.Vec3d(0.0, 0.0, 0.25)


def _slug(value: str) -> str:
    """Convert a config or prim path to a stable file-name component."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _golden_filename(config_path: str, render_product_path: str) -> str:
    """Build a short, readable golden filename that remains unique per sensor render product."""
    render_product_name = re.sub(r"(?i)(?:_?render_?product)$", "_rp", Path(render_product_path).name)
    return f"{_slug(Path(config_path).stem)}__{_slug(render_product_name)}.png"


def _set_depth_camera_view(camera_path: str) -> None:
    """Set a deterministic world-space view for a potentially parented camera."""
    stage = stage_utils.get_current_stage(backend="usd")
    camera_prim = stage.GetPrimAtPath(camera_path)
    parent_transform = UsdGeom.Xformable(camera_prim).ComputeParentToWorldTransform(Usd.TimeCode.Default())
    parent_scale = np.asarray(Gf.Transform(parent_transform).GetScale())
    ViewportManager.set_camera_view(
        camera_path,
        eye=(np.asarray(_CAMERA_EYE) * parent_scale).tolist(),
        target=(np.asarray(_CAMERA_TARGET) * parent_scale).tolist(),
    )


def _configure_fallback_depth_sensor(sensor: SingleViewDepthCameraSensor) -> None:
    """Apply deterministic depth settings to a generated render product."""
    sensor.set_sensor_baseline(55.0)
    sensor.set_sensor_focal_length(891.0)
    sensor.set_sensor_size(1280.0)
    sensor.set_sensor_maximum_disparity(110.0)
    sensor.set_sensor_disparity_confidence(0.99)
    sensor.set_sensor_noise_parameters(noise_mean=0.5, noise_sigma=1.0)
    sensor.set_sensor_disparity_noise_downscale(1.0)
    sensor.set_sensor_distance_cutoffs(minimum_distance=0.5, maximum_distance=9999.9)


def _create_depth_sensors(rtx_camera: RtxCamera) -> list[SingleViewDepthCameraSensor]:
    """Wrap authored depth products, or create a regular product for a depth camera."""
    asset_root_path = getattr(rtx_camera, "_asset_root_path", None)
    if asset_root_path is None:
        return []
    stage = stage_utils.get_current_stage(backend="usd")
    root_prim = stage.GetPrimAtPath(asset_root_path)
    if not root_prim.IsValid():
        carb.log_warn(f"Asset root prim at {asset_root_path} is not valid; skipping depth setup.")
        return []

    authored_render_products: list[tuple[Usd.Prim, str]] = []
    for child in Usd.PrimRange(root_prim):
        if not (
            child.GetTypeName() == "RenderProduct"
            and child.HasAPI("OmniSensorDepthSensorSingleViewAPI")
            and child.HasRelationship("camera")
        ):
            continue
        targets = child.GetRelationship("camera").GetTargets()
        if len(targets) != 1:
            continue
        authored_render_products.append((child, str(targets[0])))

    sensors: list[SingleViewDepthCameraSensor] = []
    for render_product, camera_path in authored_render_products:
        _set_depth_camera_view(camera_path)
        sensor = SingleViewDepthCameraSensor(camera_path, annotators="depth_sensor_distance")
        if str(sensor.render_product.GetPath()) != str(render_product.GetPath()):
            raise RuntimeError(
                f"Expected depth sensor for '{camera_path}' to attach to '{render_product.GetPath()}', "
                f"but it attached to '{sensor.render_product.GetPath()}'."
            )
        sensors.append(sensor)

    if sensors:
        return sensors

    cameras = [child for child in Usd.PrimRange(root_prim) if child.GetTypeName() == "Camera"]
    depth_cameras = [
        camera for camera in cameras if any(hint in str(camera.GetPath()).lower() for hint in ("depth", "tof"))
    ]
    if not depth_cameras:
        depth_cameras = [camera for camera in cameras if "left" in str(camera.GetPath()).lower()]
    if not depth_cameras and cameras:
        depth_cameras = cameras[:1]
    elif depth_cameras:
        depth_cameras = depth_cameras[-1:]

    for camera in depth_cameras:
        camera_path = str(camera.GetPath())
        _set_depth_camera_view(camera_path)
        sensor = SingleViewDepthCameraSensor(
            camera_path,
            resolution=_FALLBACK_DEPTH_RESOLUTION,
            annotators="depth_sensor_distance",
        )
        _configure_fallback_depth_sensor(sensor)
        sensor._source_render_product_path = (
            f"{asset_root_path}/GeneratedDepthRenderProducts/{camera.GetName()}_render_product"
        )
        sensors.append(sensor)
    return sensors


class TestDepthSensorOutput(omni.kit.test.AsyncTestCase):
    """Test depth output from all depth-enabled camera registry entries."""

    async def setUp(self) -> None:
        """Create the deterministic depth-validation scene."""
        super().setUp()
        self._retained_cameras: list[RtxCamera] = []
        self._retained_sensors: list[SingleViewDepthCameraSensor] = []

        await stage_utils.create_new_stage_async()
        await ViewportManager.wait_for_viewport_async()

        sphere_light = SphereLight("/World/SphereLight", positions=[1.0, -1.0, 1.0])
        sphere_light.set_intensities(intensities=100000)
        GroundPlane("/World/GroundPlane")
        Cone("/World/Cone", radii=0.5, heights=1.0, positions=[0.0, 0.0, 0.0], colors=[1.0, 0.0, 0.0])
        Cube("/World/Cube", sizes=0.5, positions=[-0.5, 0.25, 0.25], colors=[0.0, 1.0, 0.0])
        Sphere("/World/Sphere", radii=0.25, positions=[0.25, -0.35, 0.25], colors=[0.0, 0.0, 1.0])

    async def tearDown(self) -> None:
        """Release sensors, stop playback, and clear the stage."""
        app_utils.stop(commit=True)
        await app_utils.update_app_async()

        self._retained_sensors.clear()
        self._retained_cameras.clear()
        gc.collect()
        await app_utils.update_app_async(steps=3)
        await stage_utils.create_new_stage_async()
        super().tearDown()

    async def _run_depth_sensor_output(self, config_path: str) -> None:
        """Capture and compare every depth render product for one camera config.

        Args:
            config_path: Asset-relative path from ``SUPPORTED_CAMERA_CONFIGS``.
        """
        metadata = get_camera_metadata(config_path)
        self.assertTrue(metadata["is_depth_sensor"], f"Config is not marked as a depth sensor: {config_path}")

        rtx_camera = RtxCamera.create(
            path=stage_utils.generate_next_free_path(metadata["prim_prefix"]),
            usd_path=get_assets_root_path() + config_path,
        )
        self._retained_cameras.append(rtx_camera)
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

        sensors = _create_depth_sensors(rtx_camera)
        self._retained_sensors.extend(sensors)
        self.assertGreater(
            len(sensors),
            0,
            f"Config '{config_path}' has no usable camera for depth output.",
        )

        render_products: dict[str, tuple[str, Any]] = {}
        stage = stage_utils.get_current_stage(backend="usd")
        for sensor in sensors:
            for camera_path in sensor.camera.paths:
                self.assertTrue(
                    stage.GetPrimAtPath(camera_path).IsValid(),
                    f"Config '{config_path}' did not load camera prim '{camera_path}'.",
                )
                _set_depth_camera_view(camera_path)

            render_product_path = str(sensor.render_product.GetPath())
            render_product_name = render_product_path.rsplit("/", maxsplit=1)[-1]
            self.assertNotIn(
                render_product_name,
                render_products,
                f"Config '{config_path}' produced duplicate render-product name '{render_product_name}'.",
            )
            render_products[render_product_name] = (render_product_path, sensor)

        self.assertEqual(
            len(render_products),
            len(sensors),
            f"Config '{config_path}' did not expose every depth-sensor render product.",
        )

        for _, sensor in render_products.values():
            sensor.set_enabled_post_processing(True)
        await app_utils.update_app_async(steps=3)

        frames: dict[str, dict[str, Any]] = {}
        completed_update = None
        app_utils.play(commit=True)
        try:
            for update_index in range(1, _MAXIMUM_UPDATES + 1):
                await app_utils.update_app_async()
                current_frames: dict[str, dict[str, Any]] = {}
                for render_product_name, (_, sensor) in render_products.items():
                    data, info = sensor.get_data("depth_sensor_distance")
                    if data is None:
                        continue
                    data = data.numpy() if hasattr(data, "numpy") else data
                    array = np.asarray(data)
                    if not array.size:
                        continue
                    current_frames[render_product_name] = {
                        "data": np.array(array, copy=True),
                        "info": dict(info),
                    }
                frames = current_frames
                if update_index >= _MINIMUM_UPDATES + 5 and len(frames) == len(render_products):
                    completed_update = update_index
                    break
        finally:
            app_utils.stop(commit=True)
            await app_utils.update_app_async()

        missing_render_products = sorted(set(render_products) - set(frames))
        self.assertIsNotNone(
            completed_update,
            f"Timed out after {_MAXIMUM_UPDATES} updates waiting for native depth-sensor output from "
            f"config '{config_path}'. Missing render products: {missing_render_products}.",
        )
        assert completed_update is not None
        self.assertGreaterEqual(completed_update, _MINIMUM_UPDATES)

        for render_product_name, (render_product_path, sensor) in render_products.items():
            frame = frames[render_product_name]
            selected_annotator = "depth_sensor_distance"
            raw = frame["data"]
            height, width = sensor.resolution
            self.assertEqual(
                raw.size,
                height * width,
                f"Config '{config_path}', render product '{render_product_path}' returned "
                f"{raw.size} depth values for resolution {(height, width)}.",
            )
            raw = raw.reshape((height, width, 1))

            finite_values = raw[np.isfinite(raw)]
            self.assertGreater(
                finite_values.size,
                0,
                f"Config '{config_path}', render product '{render_product_path}' returned no finite depth values.",
            )
            self.assertGreater(
                np.ptp(finite_values),
                0.0,
                f"Config '{config_path}', render product '{render_product_path}' returned constant depth values.",
            )

            image = draw_annotator_data_to_image(
                annotator=selected_annotator,
                data=raw,
                info=frame["info"],
            )
            self.assertEqual(image.ndim, 2)
            self.assertEqual(image.dtype, np.uint8)

            stable_render_product_path = getattr(sensor, "_source_render_product_path", render_product_path)
            filename = _golden_filename(config_path, stable_render_product_path)
            golden_path = _GOLDEN_DIR / filename

            if SAVE_DEBUG_IMAGES:
                _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                debug_path = _DEBUG_DIR / filename
                self.assertTrue(cv2.imwrite(str(debug_path), image), f"Failed to save debug image '{debug_path}'.")

            if UPDATE_GOLDEN_IMAGES:
                _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
                self.assertTrue(cv2.imwrite(str(golden_path), image), f"Failed to save golden image '{golden_path}'.")
                carb.log_warn(f"Updated depth-sensor golden image at: {golden_path}")
                continue

            self.assertTrue(
                golden_path.exists(),
                f"Golden image not found for config '{config_path}', render product '{render_product_path}' at "
                f"'{golden_path}'. Set UPDATE_GOLDEN_IMAGES = True to generate it.",
            )
            golden = cv2.imread(str(golden_path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(
                golden,
                f"Failed to load golden image for config '{config_path}', render product "
                f"'{render_product_path}' from '{golden_path}'.",
            )
            self.assertEqual(
                golden.shape,
                image.shape,
                f"Golden shape mismatch for config '{config_path}', render product '{render_product_path}': "
                f"{golden.shape} != {image.shape}.",
            )

            result = compare_arrays_within_tolerances(
                golden_array=golden,
                test_array=image,
                allclose_rtol=None,
                allclose_atol=None,
                mean_tolerance=_MEAN_TOLERANCE,
                print_all_stats=True,
            )
            self.assertTrue(
                result["passed"],
                f"Golden comparison failed for config '{config_path}', render product '{render_product_path}' "
                f"with mean tolerance {_MEAN_TOLERANCE}. Statistics: {result}.",
            )


def _make_depth_sensor_test(config_path: str) -> Any:
    """Create a dynamically named async test for one depth-sensor config."""

    async def test_depth_sensor_output(self: TestDepthSensorOutput) -> None:
        await self._run_depth_sensor_output(config_path)

    test_depth_sensor_output.__doc__ = f"Test depth-sensor output for ``{config_path}``."
    return test_depth_sensor_output


def _install_depth_sensor_tests() -> None:
    """Install one dynamic test for each registered depth-sensor asset."""
    for config_path in SUPPORTED_CAMERA_CONFIGS:
        # Temporarily exclude OAK-D ToF pending upstream Metrics Assembler integration changes.
        if config_path == "/Isaac/Sensors/Luxonis/OAK-D_ToF/oak_d_tof.usd":
            continue
        if get_camera_metadata(config_path)["is_depth_sensor"]:
            test_name = f"test_depth_sensor_output__{_slug(config_path)}"
            test = _make_depth_sensor_test(config_path)
            test.__name__ = test_name
            setattr(TestDepthSensorOutput, test_name, test)


_install_depth_sensor_tests()
del _install_depth_sensor_tests
