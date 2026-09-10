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

import traceback
from typing import Any

import carb
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
import omni.syntheticdata._syntheticdata as sd
from isaacsim.core.nodes import BaseWriterNode

# No isaacsim.core.experimental wrapper exists for Usd.EditContext; direct pxr usage is intentional.
from pxr import Usd


class OgnZMQCameraHelperInternalState(BaseWriterNode):
    def __init__(self) -> None:
        self.rv_rgb = ""
        self.rv_depth = ""
        self.resetSimulationTimeOnStop = True
        self.publishStepSize = 1
        super().__init__(initialize=False)

    def post_attach(self, writer: rep.Writer, render_product: str) -> None:
        sdg = omni.syntheticdata.SyntheticData.Get()
        # The simulation gate defaults to step=1 (publish every frame), so only touch it
        # when actually skipping frames. This also avoids a spurious "invalid node" warning:
        # the depth annotator's gate node may not be in the SDG graph yet at post_attach.
        if self.publishStepSize != 1:
            for rv in (self.rv_rgb, self.rv_depth):
                if not rv:
                    continue
                try:
                    sdg.set_node_attributes(
                        rv + "IsaacSimulationGate", {"inputs:step": self.publishStepSize}, render_product
                    )
                except Exception as e:  # noqa: BLE001 — gate node may not exist yet
                    carb.log_warn(f"ZMQCameraHelper: could not set step on {rv}IsaacSimulationGate: {e}")
        # Set independently so a not-yet-ready gate above can't skip this.
        try:
            sdg.set_node_attributes(
                "IsaacReadSimulationTime", {"inputs:resetOnStop": self.resetSimulationTimeOnStop}, render_product
            )
        except Exception as e:  # noqa: BLE001
            carb.log_warn(f"ZMQCameraHelper: could not set IsaacReadSimulationTime.resetOnStop: {e}")


class OgnZMQCameraHelper:
    @staticmethod
    def internal_state() -> OgnZMQCameraHelperInternalState:
        return OgnZMQCameraHelperInternalState()

    @staticmethod
    def compute(db: Any) -> bool:
        if db.per_instance_state.initialized is False:
            db.per_instance_state.initialized = True
            stage = omni.usd.get_context().get_stage()
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                render_product_path = db.inputs.renderProductPath
                if not render_product_path:
                    carb.log_warn("ZMQCameraHelper: render product path not set")
                    db.per_instance_state.initialized = False
                    return False
                if not stage.GetPrimAtPath(render_product_path).IsValid():
                    carb.log_warn("ZMQCameraHelper: render product not created yet, retrying")
                    db.per_instance_state.initialized = False
                    return False

                db.per_instance_state.resetSimulationTimeOnStop = db.inputs.resetSimulationTimeOnStop
                db.per_instance_state.publishStepSize = db.inputs.frameSkipCount + 1

                if not db.inputs.publishRgb and not db.inputs.publishDepth:
                    carb.log_warn("ZMQCameraHelper: publishRgb and publishDepth are both disabled; nothing to publish")

                if db.inputs.useSystemTime and db.inputs.resetSimulationTimeOnStop:
                    carb.log_warn("ZMQCameraHelper: system time used, ignoring resetSimulationTimeOnStop")

                time_type = "SystemTime" if db.inputs.useSystemTime else ""

                ip = db.inputs.ip
                port = db.inputs.port
                use_ipc = db.inputs.useIpc

                try:
                    # RGB writer
                    if db.inputs.publishRgb:
                        db.per_instance_state.rv_rgb = (
                            omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
                        )
                        writer_rgb = rep.writers.get(db.per_instance_state.rv_rgb + f"ZMQ{time_type}PublishImage")
                        if writer_rgb is None:
                            carb.log_error(
                                f"ZMQCameraHelper: RGB writer "
                                f"'{db.per_instance_state.rv_rgb}ZMQ{time_type}PublishImage' is not registered; "
                                "cannot publish RGB"
                            )
                            db.per_instance_state.reset()  # clear partial writers + allow retry
                            return False
                        # IsaacConvertRGBAToRGB emits 3-channel rgb8.
                        writer_rgb.initialize(ip=ip, port=port, encoding="rgb8", useIpc=use_ipc)
                        db.per_instance_state.append_writer(writer_rgb)

                    # Depth writer (DistanceToImagePlane has the passthrough-ptr annotator)
                    if db.inputs.publishDepth:
                        db.per_instance_state.rv_depth = (
                            omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                sd.SensorType.DistanceToImagePlane.name
                            )
                        )
                        writer_depth = rep.writers.get(db.per_instance_state.rv_depth + f"ZMQ{time_type}PublishImage")
                        if writer_depth is None:
                            carb.log_error(
                                f"ZMQCameraHelper: depth writer "
                                f"'{db.per_instance_state.rv_depth}ZMQ{time_type}PublishImage' is not registered; "
                                "cannot publish depth"
                            )
                            db.per_instance_state.reset()  # clear partial writers + allow retry
                            return False
                        writer_depth.initialize(ip=ip, port=port, encoding="32FC1", useIpc=use_ipc)
                        db.per_instance_state.append_writer(writer_depth)

                    db.per_instance_state.attach_writers(render_product_path)

                except Exception:
                    carb.log_error(f"ZMQCameraHelper: failed to set up writers:\n{traceback.format_exc()}")
                    db.per_instance_state.reset()  # roll back partial attach + allow retry
                    return False

        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True

    @staticmethod
    def release_instance(node: Any, graph_instance_id: Any) -> None:
        try:
            state = OgnZMQCameraHelperInternalState.per_instance_internal_state(node)
        except Exception:
            state = None

        if state is not None:
            state.reset()
