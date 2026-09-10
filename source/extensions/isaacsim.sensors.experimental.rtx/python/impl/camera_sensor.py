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

"""High level class for creating, wrapping and operating single camera sensors with configurable annotators."""

from __future__ import annotations

from typing import Any, Literal

import carb
import isaacsim.core.experimental.utils.prim as prim_utils
import omni.replicator.core as rep
import warp as wp
from pxr import Usd

from ._camera_common import CAMERA_ANNOTATOR_SPEC
from ._sensor_base import SensorRuntime
from .rtx_camera import RtxCamera

ANNOTATOR = Literal[
    "bounding_box_2d_loose",
    "bounding_box_2d_tight",
    "bounding_box_3d",
    "distance_to_camera",
    "distance_to_image_plane",
    "instance_id_segmentation",
    "instance_segmentation",
    "motion_vectors",
    "normals",
    "pointcloud",
    "rgb",
    "rgba",
    "semantic_segmentation",
]

# Annotators that return non-image data (no reshape to resolution).
_PASSTHROUGH_ANNOTATORS = frozenset({"bounding_box_2d_tight", "bounding_box_2d_loose", "bounding_box_3d", "pointcloud"})
_RTX_POST_AA_SCHEMA = "OmniRtxPostDebugSettingsAPI_1"
_RTX_POST_AA_OP_ATTR = "omni:rtx:post:aa:op"
_RTX_POST_AA_OFF_TOKEN = "none"
_RTX_POST_AA_MIN_RESOLUTION = 300


class CameraSensor(SensorRuntime):
    """High level class for creating/wrapping and operating single camera sensor.

    Args:
        path: :class:`RtxCamera` object or single path to existing or non-existing USD Camera prim.
            If a string path is provided, a :class:`RtxCamera` instance is created internally.
        resolution: Resolution of the sensor (following OpenCV/NumPy convention: ``(height, width)``).
        annotators: Annotator/sensor types to configure.
        annotator_init_params: Per-annotator initialization parameters forwarded to Replicator annotators.
            Semantic filtering is the exception: ``semanticTypes``/``semanticFilter`` applies to the whole
            render product, so bounding box and segmentation annotators sharing one render product cannot
            be filtered independently.
        writers: Writer types to attach.
        render_vars: Render variables to pass to the render product.

    Raises:
        ValueError: If no prim is found matching the specified path.
        ValueError: If the input argument refers to more than one camera prim.
        ValueError: If an unsupported annotator type is specified.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.app as app_utils
        >>> from isaacsim.sensors.experimental.rtx import CameraSensor
        >>>
        >>> # given a USD stage with the Camera prim: /World/prim_0
        >>> resolution = (240, 320)  # following OpenCV/NumPy convention `(height, width)`
        >>> camera_sensor = CameraSensor(
        ...     "/World/prim_0",
        ...     resolution=resolution,
        ...     annotators=["rgb", "distance_to_image_plane"],
        ... )  # doctest: +NO_CHECK
        >>>
        >>> # play the simulation so the sensor can fetch data
        >>> app_utils.play(commit=True)
    """

    _AUTHORING_CLASS = RtxCamera
    _AUTHORING_ATTR = "_rtx_camera"

    def __init__(
        self,
        path: str | RtxCamera,
        *,
        resolution: tuple[int, int] | None = None,
        annotators: ANNOTATOR | list[ANNOTATOR] | None = None,
        annotator_init_params: dict[str, dict[str, Any]] | None = None,
        writers: str | list[str] | None = None,
        render_vars: list[str] | None = None,
    ) -> None:
        self._resolution = resolution
        # Set camera-specific annotator spec before super().__init__ validates annotators.
        # Subclasses (e.g. SingleViewDepthCameraSensor) may set this first with an extended spec.
        if not hasattr(self, "_annotators_spec"):
            self._annotators_spec = {k: v for k, v in CAMERA_ANNOTATOR_SPEC.items() if k in ANNOTATOR.__args__}
        super().__init__(
            path,
            annotators=annotators,
            annotator_init_params=annotator_init_params,
            writers=writers,
            render_vars=render_vars,
        )
        # Enforce square pixels on the underlying Camera prim.
        # _resolution may have been updated by _on_asset_render_product_found during super().__init__.
        if self._resolution is not None:
            self.authoring_object.camera.enforce_square_pixels(self._resolution, modes="horizontal")

    @property
    def camera(self) -> Any:
        """Camera object for accessing optical parameters.

        Returns:
            Camera object wrapping the sensor prim.
        """
        return self.authoring_object.camera

    @property
    def resolution(self) -> tuple[int, int]:
        """Resolution of the sensor.

        Returns:
            Resolution of sensor frames (following OpenCV/NumPy convention: ``(height, width)``).
        """
        return self._resolution

    def get_data(self, annotator: str, *, out: wp.array | None = None) -> tuple[wp.array | None, dict[str, Any]]:
        """Fetch the specified annotator/sensor data for the camera.

        Args:
            annotator: Annotator/sensor type from which fetch the data.
            out: Pre-allocated array to fill with the fetched data.

        Returns:
            Two-elements tuple. 1) Array containing the fetched data.
            If no data is available at the moment of calling the method, ``None`` is returned
            (annotator warm-up); use :meth:`has_data` to bound the wait. 2) Dictionary containing
            additional information according to the requested annotator/sensor. Any information
            reported alongside an empty warm-up payload is preserved.

        Raises:
            ValueError: If the specified annotator is not supported.
            ValueError: If the specified annotator is not configured.
        """
        self._validate_annotators(annotator)
        if annotator not in self._annotators:
            raise ValueError(f"The annotator '{annotator}' was not configured. Enable it when instantiating the class")
        data = self._annotators[annotator].get_data(device=str(out.device) if out is not None else "cuda")
        if isinstance(data, dict):
            info = data["info"]
            data = data["data"]
        else:
            info = {}
        if not self._record_fetched_annotator_data(data):
            return None, info
        if annotator in _PASSTHROUGH_ANNOTATORS:
            info["resolution"] = self._resolution
            return data, info
        spec = self._get_annotator_spec(annotator)
        input_channels = spec["channels"]
        output_channels = spec.get("output_channels", input_channels)
        actual_size = int(data.size)
        expected_size = int(self._resolution[0] * self._resolution[1] * input_channels)
        if actual_size != expected_size:
            raise RuntimeError(
                f"Annotator '{annotator}' returned {actual_size} elements, expected {expected_size} for "
                f"CameraSensor resolution {self._resolution} with {input_channels} input channel(s). "
                f"Render product: '{self._hydra_texture.path}'."
            )
        # Annotators attached on the host Replicator pipeline may return data
        # as a numpy.ndarray when a CUDA device is requested. Promote to a
        # Warp array on the requested device so reshape/slice/wp.copy work
        # uniformly regardless of attach device.
        if not isinstance(data, wp.array):
            target_device = out.device if out is not None else "cuda"
            data = wp.array(data, dtype=spec["dtype"], device=target_device)
        data = data.reshape((*self._resolution, input_channels))
        if out is None:
            out = data[:, :, :output_channels] if "output_channels" in spec else data
        else:
            wp.copy(out, data[:, :, :output_channels] if "output_channels" in spec else data)
        return out, info

    def _on_asset_render_product_found(self, render_product_prim: Usd.Prim) -> None:
        """Adopt the pre-authored render product's resolution in place of any requested resolution.

        Log a warning when the requested resolution differs from the authored value before replacing it.

        Args:
            render_product_prim: Pre-authored render product prim.
        """
        authored_resolution = render_product_prim.GetAttribute("resolution").Get()
        rp_resolution = (int(authored_resolution[1]), int(authored_resolution[0]))
        if self._resolution is not None and tuple(self._resolution) != rp_resolution:
            carb.log_warn(
                f"Requested resolution {tuple(self._resolution)} differs from the asset render product's "
                f"authored resolution {rp_resolution}; using the asset's resolution."
            )
        self._resolution = rp_resolution

    def _create_render_product_and_attach(
        self,
        annotators: str | list[str],
        *,
        render_vars: list[str] | None = None,
        annotator_init_params: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Create a resolution-aware render product and attach annotators.

        Args:
            annotators: Annotators to attach to the render product.
            render_vars: Render variable names to author, or ``None`` to use Replicator defaults.
            annotator_init_params: Per-annotator initialization parameters forwarded to Replicator annotators.
                Semantic filtering is the exception: ``semanticTypes``/``semanticFilter`` applies to the whole
                render product, so bounding box and segmentation annotators sharing one render product cannot
                be filtered independently.

        Raises:
            ValueError: If no sensor resolution is configured.
        """
        if self._resolution is None:
            raise ValueError("'resolution' is required when creating a new render product.")
        self._hydra_texture = rep.create.render_product(
            camera=self.authoring_object.paths[0],
            resolution=(self._resolution[1], self._resolution[0]),  # (width, height)
            name=f"camera_sensor_{hash(self)}",
            render_vars=render_vars,
        )
        self._disable_low_resolution_render_product_post_aa()
        self.attach_annotators(annotators, annotator_init_params=annotator_init_params)

    def _disable_low_resolution_render_product_post_aa(self) -> None:
        """Disable post anti-aliasing on low-resolution sensor-owned render products."""
        if min(self._resolution) >= _RTX_POST_AA_MIN_RESOLUTION:
            return
        render_product_prim = prim_utils.get_prim_at_path(self._hydra_texture.path)
        if not render_product_prim.IsValid():
            carb.log_warn(f"Unable to configure render settings for render product '{self._hydra_texture.path}'.")
            return
        if _RTX_POST_AA_SCHEMA not in render_product_prim.GetAppliedSchemas():
            render_product_prim.ApplyAPI(_RTX_POST_AA_SCHEMA)
        render_product_prim.GetAttribute(_RTX_POST_AA_OP_ATTR).Set(_RTX_POST_AA_OFF_TOKEN)
        carb.log_warn(
            f"Disabled post anti-aliasing for low-resolution CameraSensor render product "
            f"'{self._hydra_texture.path}' at resolution {self._resolution}. RTX post-AA/DLSS requires input "
            f"dimensions of at least {_RTX_POST_AA_MIN_RESOLUTION} pixels and can return buffers that do not match "
            f"the authored sensor resolution below that floor."
        )
