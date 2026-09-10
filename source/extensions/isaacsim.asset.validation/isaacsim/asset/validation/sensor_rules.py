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

"""Validation rules for authored sensor USD assets."""

from __future__ import annotations

import math

import usd_validation_nvidia
from pxr import Sdf, Usd, UsdGeom, UsdRender

__all__ = ["RGBSensorUsdRule"]

_OMNI_SENSOR_API = "OmniSensorAPI"
_TICK_RATE_ATTRIBUTE = "omni:sensor:tickRate"
_RGB_RENDER_VAR = "LdrColor"


@usd_validation_nvidia.register_rule("IsaacSim.SensorRules")
class RGBSensorUsdRule(usd_validation_nvidia.BaseRuleChecker):
    """Validate the structure and connections of an authored RGB sensor USD asset.

    The rule expects a top-level default ``Xform`` containing exactly one Camera
    and one RenderProduct. The Camera must have ``OmniSensorAPI`` and a
    nonnegative tick rate. The RenderProduct must have a positive resolution,
    target that Camera, and include an ``LdrColor`` RenderVar in ``orderedVars``.
    """

    def CheckStage(self, stage: Usd.Stage) -> None:  # noqa: N802
        """Validate the sensor asset root.

        Args:
            stage: USD stage containing the RGB sensor asset.
        """
        root = stage.GetDefaultPrim()
        if not root:
            self._AddFailedCheck(message="RGB sensor asset does not have a default prim", at=stage)
            return

        if root.GetPath().GetParentPath() != Sdf.Path.absoluteRootPath:
            self._AddFailedCheck(message=f"RGB sensor default prim {root.GetPath()} must be top-level", at=root)

        if not root.IsA(UsdGeom.Xform):
            self._AddFailedCheck(message=f"RGB sensor default prim {root.GetPath()} must be an Xform", at=root)

        camera_found = False
        render_product_found = False
        for candidate in Usd.PrimRange(root):
            if candidate.IsA(UsdGeom.Camera):
                if camera_found:
                    self._AddFailedCheck(
                        message=f"RGB sensor asset contains unexpected additional Camera {candidate.GetPath()}",
                        at=candidate,
                    )
                camera_found = True
            elif candidate.IsA(UsdRender.Product):
                if render_product_found:
                    self._AddFailedCheck(
                        message=f"RGB sensor asset contains unexpected additional RenderProduct {candidate.GetPath()}",
                        at=candidate,
                    )
                render_product_found = True

        if not camera_found:
            self._AddFailedCheck(
                message="RGB sensor asset must contain exactly one Camera; found 0",
                at=root,
            )
        if not render_product_found:
            self._AddFailedCheck(
                message="RGB sensor asset must contain exactly one RenderProduct; found 0",
                at=root,
            )

    def CheckPrim(self, prim: Usd.Prim) -> None:  # noqa: N802
        """Validate the resolved RGB sensor hierarchy for each variant selection.

        Args:
            prim: USD prim visited by the validation engine.
        """
        stage = prim.GetStage()
        root = stage.GetDefaultPrim()
        if not root or not prim.GetPath().HasPrefix(root.GetPath()):
            return

        if prim.IsA(UsdGeom.Camera):
            self._check_camera(prim)
        elif prim.IsA(UsdRender.Product):
            self._check_render_product(prim, root)

    def _check_camera(self, camera: Usd.Prim) -> None:
        """Validate the RGB camera schema and tick rate.

        Args:
            camera: Camera prim to validate.
        """
        if not camera.HasAPI(_OMNI_SENSOR_API):
            self._AddFailedCheck(message=f"Camera {camera.GetPath()} does not have OmniSensorAPI", at=camera)

        tick_rate_attr = camera.GetAttribute(_TICK_RATE_ATTRIBUTE)
        tick_rate = tick_rate_attr.Get() if tick_rate_attr else None
        if tick_rate is None or not math.isfinite(tick_rate) or tick_rate < 0:
            self._AddFailedCheck(
                message=f"Camera {camera.GetPath()} must have a finite, nonnegative {_TICK_RATE_ATTRIBUTE}",
                at=tick_rate_attr if tick_rate_attr else camera,
            )

    def _check_render_product(self, render_product_prim: Usd.Prim, root: Usd.Prim) -> None:
        """Validate the render product resolution and relationships.

        Args:
            render_product_prim: RenderProduct prim to validate.
            root: The validated RGB sensor asset root.
        """
        render_product = UsdRender.Product(render_product_prim)
        resolution_attr = render_product.GetResolutionAttr()
        width, height = resolution_attr.Get()
        if width <= 0 or height <= 0:
            self._AddFailedCheck(
                message=f"RenderProduct {render_product_prim.GetPath()} must have a positive width and height",
                at=resolution_attr,
            )

        camera_rel = render_product.GetCameraRel()
        camera_targets = camera_rel.GetTargets() if camera_rel else []
        if len(camera_targets) != 1:
            self._AddFailedCheck(
                message=f"RenderProduct {render_product_prim.GetPath()} must target exactly one Camera",
                at=camera_rel if camera_rel else render_product_prim,
            )
        else:
            camera = render_product_prim.GetStage().GetPrimAtPath(camera_targets[0])
            if not camera or not camera.IsA(UsdGeom.Camera) or not camera.GetPath().HasPrefix(root.GetPath()):
                self._AddFailedCheck(
                    message=(
                        f"RenderProduct {render_product_prim.GetPath()} camera target {camera_targets[0]} "
                        f"must resolve to the Camera under {root.GetPath()}"
                    ),
                    at=camera_rel,
                )

        self._check_ordered_vars(render_product)

    def _check_ordered_vars(self, render_product: UsdRender.Product) -> None:
        """Validate that orderedVars contains a valid LdrColor RenderVar.

        Args:
            render_product: RenderProduct schema object to validate.
        """
        render_product_prim = render_product.GetPrim()
        ordered_vars_rel = render_product.GetOrderedVarsRel()
        ordered_var_targets = ordered_vars_rel.GetTargets() if ordered_vars_rel else []
        if not ordered_var_targets:
            self._AddFailedCheck(
                message=f"RenderProduct {render_product_prim.GetPath()} must have orderedVars targets",
                at=ordered_vars_rel if ordered_vars_rel else render_product_prim,
            )
            return

        stage = render_product_prim.GetStage()
        has_rgb_render_var = False
        for target in ordered_var_targets:
            render_var_prim = stage.GetPrimAtPath(target)
            if not render_var_prim or not render_var_prim.IsA(UsdRender.Var):
                self._AddFailedCheck(
                    message=f"orderedVars target {target} is not a valid RenderVar", at=ordered_vars_rel
                )
                continue
            source_name = UsdRender.Var(render_var_prim).GetSourceNameAttr().Get()
            if str(source_name) == _RGB_RENDER_VAR:
                has_rgb_render_var = True

        if not has_rgb_render_var:
            self._AddFailedCheck(
                message=f"RenderProduct {render_product_prim.GetPath()} orderedVars must include an LdrColor RenderVar",
                at=ordered_vars_rel,
            )
