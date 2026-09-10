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

"""Tests for RGB sensor USD validation."""

from __future__ import annotations

import omni.kit.test
import usd_validation_nvidia
from isaacsim.asset.validation.sensor_rules import RGBSensorUsdRule
from pxr import Gf, Sdf, Usd, UsdGeom, UsdRender

_ROOT_PATH = Sdf.Path("/CameraSensor")
_CAMERA_PATH = _ROOT_PATH.AppendChild("CameraRGB")
_RENDER_PRODUCT_PATH = _ROOT_PATH.AppendChild("RenderProduct")
_RENDER_VAR_PATH = _RENDER_PRODUCT_PATH.AppendChild("RenderVar")


def _build_rgb_sensor_stage(root_path: Sdf.Path = _ROOT_PATH) -> Usd.Stage:
    """Build a valid in-memory RGB sensor USD stage.

    Args:
        root_path: Path of the sensor root Xform.

    Returns:
        Stage with a complete RGB sensor asset under ``root_path``.
    """
    stage = Usd.Stage.CreateInMemory()
    parent_path = root_path.GetParentPath()
    if parent_path != Sdf.Path.absoluteRootPath:
        UsdGeom.Xform.Define(stage, parent_path)
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())

    camera_path = root_path.AppendChild("CameraRGB")
    camera = UsdGeom.Camera.Define(stage, camera_path).GetPrim()
    camera.ApplyAPI("OmniSensorAPI")
    camera.CreateAttribute("omni:sensor:tickRate", Sdf.ValueTypeNames.Float).Set(60.0)

    render_product_path = root_path.AppendChild("RenderProduct")
    render_product = UsdRender.Product.Define(stage, render_product_path)
    render_product.CreateResolutionAttr(Gf.Vec2i(1920, 1080))
    render_product.CreateCameraRel().SetTargets([camera_path])

    render_var_path = render_product_path.AppendChild("RenderVar")
    render_var = UsdRender.Var.Define(stage, render_var_path)
    render_var.CreateSourceNameAttr("LdrColor")
    render_product.CreateOrderedVarsRel().SetTargets([render_var_path])
    return stage


def _validate(stage: Usd.Stage) -> list[str]:
    """Return failure messages emitted by RGBSensorUsdRule.

    Args:
        stage: Stage to validate.

    Returns:
        Failure and error messages from the rule.
    """
    engine = usd_validation_nvidia.ValidationEngine(init_rules=False, variants=True)
    engine.enable_rule(RGBSensorUsdRule)
    results = engine.validate(stage)
    return [
        issue.message
        for issue in results
        if issue.severity in (usd_validation_nvidia.IssueSeverity.FAILURE, usd_validation_nvidia.IssueSeverity.ERROR)
    ]


class TestRGBSensorUsdRule(omni.kit.test.AsyncTestCase):
    """Validate the base RGB sensor USD contract."""

    async def test_valid_rgb_sensor_passes(self) -> None:
        """A complete RGB sensor asset passes validation."""
        self.assertEqual(_validate(_build_rgb_sensor_stage()), [])

    async def test_stage_requires_default_prim(self) -> None:
        """The sensor asset must define a default prim."""
        stage = Usd.Stage.CreateInMemory()

        self.assertEqual(_validate(stage), ["RGB sensor asset does not have a default prim"])

    async def test_default_prim_must_be_top_level(self) -> None:
        """The sensor asset default prim must be top-level."""
        root_path = Sdf.Path("/Parent/CameraSensor")
        stage = _build_rgb_sensor_stage(root_path)
        self.assertEqual(stage.GetDefaultPrim().GetPath(), root_path)

        self.assertEqual(
            _validate(stage),
            ["RGB sensor default prim /Parent/CameraSensor must be top-level"],
        )

    async def test_default_prim_must_be_xform(self) -> None:
        """The sensor asset default prim must be an Xform."""
        stage = _build_rgb_sensor_stage()
        root = stage.GetDefaultPrim()
        root.SetTypeName("Scope")

        self.assertEqual(_validate(stage), ["RGB sensor default prim /CameraSensor must be an Xform"])

    async def test_validation_accumulates_independent_failures(self) -> None:
        """Root and descendant failures are reported together."""
        stage = _build_rgb_sensor_stage()
        stage.GetDefaultPrim().SetTypeName("Scope")
        stage.GetPrimAtPath(_CAMERA_PATH).RemoveAPI("OmniSensorAPI")

        self.assertEqual(
            _validate(stage),
            [
                "RGB sensor default prim /CameraSensor must be an Xform",
                "Camera /CameraSensor/CameraRGB does not have OmniSensorAPI",
            ],
        )

    async def test_additional_sensor_prims_are_reported_at_their_paths(self) -> None:
        """Each additional Camera and RenderProduct is identified directly."""
        stage = _build_rgb_sensor_stage()
        camera_path = _ROOT_PATH.AppendChild("UnexpectedCamera")
        camera = UsdGeom.Camera.Define(stage, camera_path).GetPrim()
        camera.ApplyAPI("OmniSensorAPI")
        camera.CreateAttribute("omni:sensor:tickRate", Sdf.ValueTypeNames.Float).Set(60.0)

        render_product_path = _ROOT_PATH.AppendChild("UnexpectedRenderProduct")
        render_product = UsdRender.Product.Define(stage, render_product_path)
        render_product.CreateResolutionAttr(Gf.Vec2i(1920, 1080))
        render_product.CreateCameraRel().SetTargets([_CAMERA_PATH])
        render_product.CreateOrderedVarsRel().SetTargets([_RENDER_VAR_PATH])

        self.assertEqual(
            _validate(stage),
            [
                "RGB sensor asset contains unexpected additional Camera /CameraSensor/UnexpectedCamera",
                "RGB sensor asset contains unexpected additional RenderProduct /CameraSensor/UnexpectedRenderProduct",
            ],
        )

    async def test_camera_requires_sensor_api_and_valid_tick_rate(self) -> None:
        """The Camera must carry OmniSensorAPI and a finite nonnegative tick rate."""
        stage = _build_rgb_sensor_stage()
        camera = stage.GetPrimAtPath(_CAMERA_PATH)
        camera.RemoveAPI("OmniSensorAPI")
        camera.GetAttribute("omni:sensor:tickRate").Set(-1.0)

        messages = _validate(stage)
        self.assertTrue(any("does not have OmniSensorAPI" in message for message in messages), messages)
        self.assertTrue(any("finite, nonnegative" in message for message in messages), messages)

    async def test_render_product_requires_valid_resolution_and_camera_target(self) -> None:
        """The RenderProduct must have positive dimensions and target the sensor Camera."""
        stage = _build_rgb_sensor_stage()
        render_product = UsdRender.Product(stage.GetPrimAtPath(_RENDER_PRODUCT_PATH))
        render_product.GetResolutionAttr().Set(Gf.Vec2i(0, 1080))
        render_product.GetCameraRel().SetTargets([Sdf.Path("/MissingCamera")])

        messages = _validate(stage)
        self.assertTrue(any("positive width and height" in message for message in messages), messages)
        self.assertTrue(any("must resolve to the Camera" in message for message in messages), messages)

    async def test_ordered_vars_requires_ldr_color_render_var(self) -> None:
        """The RenderProduct must wire a valid LdrColor RenderVar through orderedVars."""
        stage = _build_rgb_sensor_stage()
        render_var = UsdRender.Var(stage.GetPrimAtPath(_RENDER_VAR_PATH))
        render_var.GetSourceNameAttr().Set("Depth")

        messages = _validate(stage)
        self.assertTrue(any("LdrColor RenderVar" in message for message in messages), messages)

    async def test_invalid_sensor_config_variant_is_reported(self) -> None:
        """ValidationEngine variant traversal checks every resolved sensor configuration."""
        stage = _build_rgb_sensor_stage()
        root = stage.GetDefaultPrim()
        render_product = UsdRender.Product(stage.GetPrimAtPath(_RENDER_PRODUCT_PATH))
        render_product.GetResolutionAttr().Clear()
        variant_set = root.GetVariantSets().AddVariantSet("SensorConfig")
        for name, resolution in (("Valid", Gf.Vec2i(1920, 1080)), ("Invalid", Gf.Vec2i(0, 720))):
            variant_set.AddVariant(name)
            variant_set.SetVariantSelection(name)
            with variant_set.GetVariantEditContext():
                render_product.GetResolutionAttr().Set(resolution)
        variant_set.SetVariantSelection("Valid")

        messages = _validate(stage)
        self.assertTrue(any("positive width and height" in message for message in messages), messages)
