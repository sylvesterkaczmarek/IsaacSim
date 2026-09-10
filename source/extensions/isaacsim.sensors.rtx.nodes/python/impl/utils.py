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

"""Helpers for visualizing RTX sensor point clouds colored by a per-point scalar field.

``register_scalar_colored_point_cloud_writer`` builds, on demand, the OmniGraph chain that
colors an extracted RTX point cloud by one of its scalar outputs (distance, intensity, ...):

    IsaacExtractRTXSensorPointCloud --(point cloud)----------------------> DebugDrawPointCloud
    IsaacExtractRTXSensorPointCloud --(<scalar>)--> IsaacMapScalarsToColors --(colors)--> DebugDrawPointCloud

Because the DebugDrawPointCloud writer is registered with two annotators, Replicator inserts a
RationalTimeSyncGate that fires it only once both the point cloud and the colors have been
produced for the same frame, keeping points and colors aligned.
"""

from __future__ import annotations

import numpy as np
import omni.syntheticdata
from isaacsim.core.nodes import register_annotator_from_node_with_telemetry, register_node_writer_with_telemetry
from omni.replicator.core import AnnotatorRegistry, WriterRegistry

_EXTRACT_ANNOTATOR = "IsaacExtractRTXSensorPointCloud"
_MAP_SCALARS_NODE = "isaacsim.core.nodes.IsaacMapScalarsToColors"
_DEBUG_DRAW_NODE = "isaacsim.util.debug_draw.DebugDrawPointCloud"
_CATEGORY = "isaacsim.sensors.experimental.rtx"

# Per-point scalar fields exposed by IsaacExtractRTXSensorPointCloud that are a single float per
# point (and thus colorable).
SCALAR_FIELDS = {
    "distance": ("distancePtr", "distanceBufferSize"),
    "intensity": ("intensityPtr", "intensityBufferSize"),
    "azimuth": ("azimuthPtr", "azimuthBufferSize"),
    "elevation": ("elevationPtr", "elevationBufferSize"),
    "radialVelocityMS": ("radialVelocityMSPtr", "radialVelocityMSBufferSize"),
}


def register_scalar_colored_point_cloud_writer(
    scalar: str = "intensity",
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    log_scale: bool = True,
    writer_name: str | None = None,
    annotator_name: str | None = None,
) -> str:
    """Register a DebugDrawPointCloud writer that colors an RTX point cloud by a scalar field.

    Builds (idempotently) the ``IsaacExtractRTXSensorPointCloud -> IsaacMapScalarsToColors ->
    DebugDrawPointCloud`` graph for the chosen ``scalar`` and returns the writer's registry name,
    ready to pass to ``LidarSensor.attach_writer(name, size=...)``.

    Args:
        scalar: Per-point scalar field to color by; one of :data:`SCALAR_FIELDS`
            (e.g. ``"distance"`` or ``"intensity"``).
        base_color: RGBA multiplier applied to the IsaacMapScalarsToColors color ramp.
        log_scale: Use logarithmic (``True``) or linear (``False``) scaling of the color ramp.
        writer_name: Optional explicit writer registry name (defaults to one derived from ``scalar``).
        annotator_name: Optional explicit color-annotator name (defaults to one derived from ``scalar``).

    Returns:
        The Replicator writer registry name to attach to the sensor.

    Raises:
        ValueError: If ``scalar`` is not a supported field.
    """
    if scalar not in SCALAR_FIELDS:
        raise ValueError(f"Unsupported scalar '{scalar}'. Supported fields: {sorted(SCALAR_FIELDS)}.")

    scalar_ptr, scalar_size = SCALAR_FIELDS[scalar]
    suffix = scalar[:1].upper() + scalar[1:]
    annotator_name = annotator_name or f"IsaacRtxSensor{suffix}ToColors"
    writer_name = writer_name or f"RtxSensorDebugDrawPointCloudColoredBy{suffix}"
    node_template = omni.syntheticdata.SyntheticData.NodeConnectionTemplate

    # Idempotent: drop any previous registration so re-running in the same process is clean.
    if writer_name in WriterRegistry.get_writers():
        WriterRegistry.unregister_writer(writer_name)
    if annotator_name in AnnotatorRegistry.get_registered_annotators():
        AnnotatorRegistry.unregister_annotator(annotator_name)

    # IsaacMapScalarsToColors: the chosen per-point scalar -> per-point RGBA colors.
    register_annotator_from_node_with_telemetry(
        name=annotator_name,
        input_rendervars=[
            node_template(
                _EXTRACT_ANNOTATOR,
                attributes_mapping={
                    "outputs:exec": "inputs:execIn",
                    f"outputs:{scalar_ptr}": "inputs:scalarPtr",
                    f"outputs:{scalar_size}": "inputs:scalarBufferSize",
                },
            ),
        ],
        node_type_id=_MAP_SCALARS_NODE,
        init_params={"baseColor": list(base_color), "logScaleMode": log_scale},
        output_data_type=np.float32,
        output_channels=4,
    )

    # DebugDrawPointCloud reads the point cloud from the extract node and the colors from the
    # map-scalars node. The two annotators make Replicator insert a frame-sync gate between them.
    register_node_writer_with_telemetry(
        name=writer_name,
        node_type_id=_DEBUG_DRAW_NODE,
        annotators=[
            node_template(
                _EXTRACT_ANNOTATOR,
                attributes_mapping={
                    "outputs:dataPtr": "inputs:dataPtr",
                    "outputs:bufferSize": "inputs:bufferSize",
                    "outputs:cudaDeviceIndex": "inputs:cudaDeviceIndex",
                    "outputs:transform": "inputs:transform",
                },
            ),
            node_template(
                annotator_name,
                attributes_mapping={
                    "outputs:colorsPtr": "inputs:colorsPtr",
                    "outputs:colorsBufferSize": "inputs:colorsBufferSize",
                },
            ),
        ],
        doTransform=True,
        testMode=False,
        category=_CATEGORY,
    )

    return writer_name
