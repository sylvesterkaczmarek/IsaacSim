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

from __future__ import annotations

import carb
import omni.ext


class IsaacZmqNodesExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        carb.log_info("ZMQ Nodes Extension starting up")

        # Acquiring the interface triggers carbOnPluginStartup(), which calls
        # INITIALIZE_OGN_NODES() and registers all C++ OmniGraph node types.
        from .bindings._zmq_nodes import acquire_interface

        self._interface = acquire_interface()

        # Register the render-var -> ZMQPublishImage writers that OgnZMQCameraHelper
        # looks up via rep.writers.get("{rv}ZMQ[SystemTime]PublishImage"). Without
        # these a graph-only / loaded-USD scene cannot publish camera images.
        try:
            self._register_image_writers()
        except Exception:
            import traceback

            carb.log_warn("ZMQ Nodes: failed to register image writers:\n" + traceback.format_exc())

        carb.log_info("ZMQ Nodes Extension started")

    @staticmethod
    def _register_image_writers() -> None:
        """Register {rendervar}ZMQ[SystemTime]PublishImage writers for RGB + depth.

        Mirrors the Isaac ROS 2 camera-publisher registration: a render-var pointer
        annotator (IsaacPassthroughImagePtr) feeds the C++ ZMQPublishImage node's
        dataPtr/width/height/bufferSize/format/cudaDeviceIndex inputs (auto-mapped by
        name), and IsaacRead{Simulation,System}Time feeds inputs:timeStamp. The
        per-instance ip/port/useIpc come from OgnZMQCameraHelper.writer.initialize(...).
        """
        import omni.replicator.core as rep
        import omni.syntheticdata
        import omni.syntheticdata._syntheticdata as sd
        from isaacsim.core.nodes.scripts.utils import register_node_writer_with_telemetry

        nct = omni.syntheticdata.SyntheticData.NodeConnectionTemplate
        existing = rep.WriterRegistry.get_writers()

        for time_read, time_out, time_label in (
            ("IsaacReadSimulationTime", "outputs:simulationTime", ""),
            ("IsaacReadSystemTime", "outputs:systemTime", "SystemTime"),
        ):
            # (sensor render var, the per-rendervar pointer annotator that core.nodes
            #  registers, encoding). RGB uses the rgba->rgb converter (3 channels);
            #  depth uses the passthrough pointer (registered for DistanceToImagePlane).
            #  Both emit dataPtr/width/height/bufferSize/format/cudaDeviceIndex, which
            #  auto-map by name to ZMQPublishImage inputs.
            for sensor, annot_suffix, encoding in (
                (sd.SensorType.Rgb.name, "IsaacConvertRGBAToRGB", "rgb8"),
                (sd.SensorType.DistanceToImagePlane.name, "IsaacPassthroughImagePtr", "32FC1"),
            ):
                rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sensor)
                name = f"{rv}ZMQ{time_label}PublishImage"
                if name in existing:
                    continue
                register_node_writer_with_telemetry(
                    name=name,
                    node_type_id="isaacsim.zmq.nodes.ZMQPublishImage",
                    annotators=[
                        f"{rv}{annot_suffix}",
                        nct(time_read, attributes_mapping={time_out: "inputs:timeStamp"}),
                    ],
                    encoding=encoding,
                    category="isaacsim.zmq.nodes",
                )

    def on_shutdown(self) -> None:
        carb.log_info("ZMQ Nodes Extension shutting down")

        from .bindings._zmq_nodes import release_interface

        if hasattr(self, "_interface") and self._interface is not None:
            release_interface(self._interface)
            self._interface = None

        carb.log_info("ZMQ Nodes Extension shut down")
