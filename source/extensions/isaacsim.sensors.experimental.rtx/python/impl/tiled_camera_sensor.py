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

"""Tiled camera sensor implementation for batch processing multiple cameras with shared annotators."""

from __future__ import annotations

from typing import Any, Literal, get_args

import carb
import omni.replicator.core as rep
import warp as wp
from isaacsim.core.experimental.objects import Camera

from ._camera_common import CAMERA_ANNOTATOR_SPEC as ANNOTATOR_SPEC
from ._sensor_base import SensorRuntime

ANNOTATOR = Literal[
    "distance_to_camera",
    "distance_to_image_plane",
    "instance_id_segmentation",
    "instance_segmentation",
    "motion_vectors",
    "normals",
    "rgb",
    "rgba",
    "semantic_segmentation",
]


class TiledCameraSensor(SensorRuntime):
    """High level class for creating/wrapping and operating tiled (batched) camera sensors.

    Args:
        paths: ``Camera`` object, single path or list of paths to existing or non-existing (one of both) USD Camera prims.
            Can include regular expressions for matching multiple prims.
        resolution: Resolution of each individual sensor (following OpenCV/NumPy convention: ``(height, width)``).
        annotators: Annotator/sensor types to configure.
        annotator_init_params: Per-annotator initialization parameters forwarded to Replicator annotators.
            Semantic filtering is the exception: ``semanticTypes``/``semanticFilter`` applies to the whole
            render product, so bounding box and segmentation annotators sharing one render product cannot
            be filtered independently.
        writers: Writer types to attach.
        render_vars: Reserved for API compatibility. Tiled render products do not support additional render variables,
            so this argument is ignored and a warning is logged when it is set.

    Raises:
        ValueError: If no prims are found matching the specified paths.
        ValueError: If an unsupported annotator type is specified.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.app as app_utils
        >>> from isaacsim.sensors.experimental.rtx import TiledCameraSensor
        >>>
        >>> # given a USD stage with the Camera prims: /World/prim_0, /World/prim_1, and /World/prim_2
        >>> resolution = (240, 320)  # following OpenCV/NumPy convention `(height, width)`
        >>> tiled_camera_sensor = TiledCameraSensor(
        ...     "/World/prim_.*",
        ...     resolution=resolution,
        ...     annotators=["rgb", "distance_to_image_plane"],
        ... )  # doctest: +NO_CHECK
        >>>
        >>> # play the simulation so the sensor can fetch data
        >>> app_utils.play(commit=True)
    """

    _AUTHORING_CLASS = Camera
    _AUTHORING_ATTR = "_camera"
    _ALLOW_MULTIPLE_AUTHORING_OBJECTS = True

    def __init__(
        self,
        paths: str | list[str] | Camera,
        *,
        resolution: tuple[int, int],
        annotators: ANNOTATOR | list[ANNOTATOR],
        annotator_init_params: dict[str, dict[str, Any]] | None = None,
        writers: str | list[str] | None = None,
        render_vars: list[str] | None = None,
    ) -> None:
        self._resolution = resolution
        self._tiled_resolution = None
        # set the camera-specific annotator spec before `super().__init__` validates annotators
        if not hasattr(self, "_annotators_spec"):
            self._annotators_spec = {annotator: ANNOTATOR_SPEC[annotator] for annotator in get_args(ANNOTATOR)}
        super().__init__(
            paths,
            annotators=annotators,
            annotator_init_params=annotator_init_params,
            writers=writers,
            render_vars=render_vars,
        )

    def __len__(self) -> int:
        """Get the number of cameras encapsulated by the sensor.

        Returns:
            Number of cameras in the sensor.

        Example:

        .. code-block:: python

            >>> len(tiled_camera_sensor)
            3
        """
        return len(self._camera)

    """
    Properties.
    """

    @property
    def camera(self) -> Camera:
        """Camera object encapsulated by the sensor.

        Returns:
            Camera object encapsulated by the sensor.

        Example:

        .. code-block:: python

            >>> tiled_camera_sensor.camera
            <isaacsim.core.experimental.objects.impl.camera.Camera object at 0x...>
        """
        return self._camera

    @property
    def resolution(self) -> tuple[int, int]:
        """Resolution of individual batched frames.

        Returns:
            Resolution of individual batched frames (following OpenCV/NumPy convention: ``(height, width)``).

        Example:

        .. code-block:: python

            >>> tiled_camera_sensor.resolution
            (240, 320)
        """
        return self._resolution

    @property
    def tiled_resolution(self) -> tuple[int, int]:
        """Resolution of tiled frames.

        Returns:
            Resolution of tiled frames (following OpenCV/NumPy convention: ``(height, width)``).

        Example:

        .. code-block:: python

            >>> tiled_camera_sensor.tiled_resolution
            (480, 640)
        """
        return self._tiled_resolution

    """
    Methods.
    """

    def get_data(
        self, annotator: str, *, tiled: bool = False, out: wp.array | None = None
    ) -> tuple[wp.array | None, dict[str, Any]]:
        """Fetch the specified annotator/sensor data for all cameras as a batch of frames or as a single tiled frame.

        Args:
            annotator: Annotator/sensor type from which fetch the data.
            tiled: Whether to get annotator/sensor data as a single tiled frame.
            out: Pre-allocated array to fill with the fetched data.

        Returns:
            Two-elements tuple. 1) Array containing the fetched data. If ``out`` is defined, such instance is returned
            filled with the data. If no data is available at the moment of calling the method, ``None`` is returned
            (annotator warm-up); use :meth:`has_data` to bound the wait. 2) Dictionary containing additional
            information according to the requested annotator/sensor. Any information reported alongside an empty
            warm-up payload is preserved.

        Raises:
            ValueError: If the specified annotator is not supported.
            ValueError: If the specified annotator is not configured when instantiating the object.

        Example:

        .. code-block:: python

            >>> data, info = tiled_camera_sensor.get_data("rgb")  # doctest: +NO_CHECK
            >>> data.shape  # doctest: +SKIP
            (3, 240, 320, 3)
            >>> info
            {}
        """
        self._validate_annotators(annotator)
        if annotator not in self._annotators:
            raise ValueError(f"The annotator '{annotator}' was not configured. Enable it when instantiating the class")
        # fetch data from annotator
        data = self._annotators[annotator].get_data(device=str(out.device) if out is not None else "cuda")
        if isinstance(data, dict):
            info = data["info"]
            data = data["data"]
        else:
            info = {}
        if not self._record_fetched_annotator_data(data):
            return None, info
        # process data
        spec = self._get_annotator_spec(annotator)
        input_channels = spec["channels"]
        output_channels = spec.get("output_channels", input_channels)
        data = data.reshape((*self._tiled_resolution, input_channels))
        # - tiled frame
        if tiled:
            if out is None:
                out = data[:, :, :output_channels] if "output_channels" in spec else data
            else:
                wp.copy(out, data[:, :, :output_channels] if "output_channels" in spec else data)
        # - batched frames
        else:
            height, width = self._resolution
            # get or create output array
            if out is None:
                shape = (len(self._camera), height, width, output_channels)
                out = wp.empty(shape, dtype=spec["dtype"], device=data.device)
            else:
                if out.device != data.device:  # move tiled data to output device if it is not the same
                    data = wp.clone(data, out.device)
            # convert tiled data to batch of frames
            wp.launch(
                kernel=_wk_reshape_tiled_image,
                dim=(len(self._camera), height, width),
                inputs=[
                    data.flatten(),
                    height,
                    width,
                    input_channels,
                    output_channels,
                    self._tiled_resolution[1] // width,
                    0,
                ],
                outputs=[out],
                device=data.device,
            )
        return out, info

    """
    Internal methods.
    """

    def _initialize_sensor(
        self,
        annotators: str | list[str],
        *,
        render_vars: list[str] | None = None,
        annotator_init_params: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize sensor by creating the tiled hydra texture and attaching annotators.

        A tiled render product is always created for the batched cameras, so this deliberately
        bypasses the pre-authored render product discovery performed by the base class.

        Args:
            annotators: Annotator/sensor types to configure.
            render_vars: Additional render variables (not supported by tiled render products; ignored).
            annotator_init_params: Per-annotator initialization parameters forwarded to Replicator annotators.
        """
        if render_vars:
            carb.log_warn(f"Tiled render products do not support additional render variables. Ignoring {render_vars}.")
        self._camera.enforce_square_pixels(self._resolution, modes="horizontal")
        # compute tiled resolution
        num_rows = round(len(self._camera) ** 0.5)
        num_columns = (len(self._camera) + num_rows - 1) // num_rows
        self._tiled_resolution = (num_rows * self._resolution[0], num_columns * self._resolution[1])  # (height, width)
        # create the hydra texture
        self._hydra_texture = rep.create.render_product_tiled(
            cameras=self._camera.paths,
            tile_resolution=(self._resolution[1], self._resolution[0]),  # (width, height)
            name=f"tiled_camera_sensor_{hash(self)}",
        )
        # attach annotators
        self.attach_annotators(annotators, annotator_init_params=annotator_init_params)


"""
Custom Warp kernels.
"""


@wp.kernel(enable_backward=False)
def _wk_reshape_tiled_image(
    tiled_data: Any,
    image_height: int,
    image_width: int,
    num_channels: int,
    num_output_channels: int,
    num_tiles_x: int,
    offset: int,
    batched_frames: Any,
) -> None:
    """Reshape a tiled data with shape ``(height * width * num_channels * num_cameras)`` to a batch of images.

    The output has shape ``(num_cameras, height, width, num_channels)``.

    Args:
        tiled_data: Tiled data with shape ``(height * width * num_channels * num_cameras)``.
        image_height: Image height.
        image_width: Image width.
        num_channels: Number of input channels.
        num_output_channels: Number of output channels.
        num_tiles_x: Number of tiles in the x direction.
        offset: Offset in the tiled data.
        batched_frames: Batch of frames with shape ``(num_cameras, height, width, num_output_channels)``.
    """
    camera_id, height_id, width_id = wp.tid()
    # resolve the tile indices
    tile_x_id = camera_id % num_tiles_x
    tile_y_id = camera_id // num_tiles_x
    # compute the pixel index in the tiled data
    pixel_index = (
        offset
        + num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)
        + num_channels * tile_x_id * image_width
        + num_channels * width_id
    )
    # copy tiled data into the batch frames
    for i in range(num_output_channels):
        batched_frames[camera_id, height_id, width_id, i] = batched_frames.dtype(tiled_data[pixel_index + i])


for dtype in [wp.uint8, wp.float32]:
    wp.overload(
        _wk_reshape_tiled_image,
        {"tiled_data": wp.array(dtype=dtype), "batched_frames": wp.array(dtype=dtype, ndim=4)},
    )
