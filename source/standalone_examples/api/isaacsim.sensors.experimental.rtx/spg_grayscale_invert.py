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

"""Author a chained RTX Sensor Processing Graph (SPG): grayscale then invert.

This example demonstrates how to:

- Author a multi-shader SPG with ``RtxCamera.author_spg`` where the output of one
  ``SPGNode`` feeds the input of the next (``GrayscaleKernel`` -> ``InvertKernel``).
- Pass a typed shader parameter (``strength``) to a kernel via ``SPGNode.params``.
- Read the final output AOV (``LdrInverted``) back as a NumPy array via a Replicator
  annotator (``register_annotator_from_aov``) — no viewport required.

The graph converts the rendered ``LdrColor`` AOV to grayscale, then inverts it. See
the SPG documentation for authoring details:
https://docs.omniverse.nvidia.com/kit/docs/omni.rtx.spg/0.2.0/Overview.html
"""

import argparse
import os

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="SPG grayscale+invert chained post-process example.")
parser.add_argument(
    "--test", default=False, action="store_true", help="Run in test mode (headless, exits after check)."
)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True})

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.replicator.core as rep
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera, SPGNode


class SingleAovWriter(rep.Writer):
    """Minimal Replicator writer that saves one AOV annotator to a PNG.

    Uses the Replicator ``Writer`` + backend I/O path instead of writing files by
    hand. The most recent frame is cached on ``last_frame`` so the example can
    validate it after stepping the orchestrator.
    """

    def __init__(self, output_dir: str, aov_name: str):
        self.version = "1.0.0"
        self._aov_name = aov_name
        self._frame_id = 0
        self.annotators = [aov_name]
        self.backend = rep.BackendDispatch(output_dir=output_dir)
        self.last_frame = None

    def write(self, data: dict):
        frame = np.asarray(data[self._aov_name])
        self.last_frame = frame
        self.backend.write_image(path=f"{self._aov_name}_{self._frame_id:04d}.png", data=frame[..., :3])
        self._frame_id += 1


app_utils.enable_extension("omni.rtx.spg")
carb.settings.get_settings().set("/rtx/spg/enabled", True)

spg_data_dir = os.path.join(app_utils.get_extension_path("isaacsim.sensors.experimental.rtx"), "data", "spg")

output_dir = os.path.join(os.getcwd(), "_example_output_isaacsim.sensors.experimental.rtx", "spg_grayscale_invert")
os.makedirs(output_dir, exist_ok=True)
resolution = (640, 360)  # (width, height)

stage_utils.open_stage(usd_path=os.path.join(spg_data_dir, "scene", "cornell_box.usda"))
while stage_utils.is_stage_loading():
    simulation_app.update()

cam = RtxCamera("/World/Camera")

# A CameraSensor creates and drives the render product for the camera. Its
# ``resolution`` is (height, width).
sensor = CameraSensor(cam, resolution=(resolution[1], resolution[0]))
render_product_path = str(sensor.render_product.GetPath())

# Chain two shaders: GrayscaleKernel.outputs:LdrGrayscale feeds InvertKernel.inputs:Image.
# InvertKernel takes a typed float parameter ``strength`` (1.0 = full inversion).
cam.author_spg(
    [
        SPGNode(
            "GrayscaleKernel",
            os.path.join(spg_data_dir, "kernels", "GrayscaleKernel.cu"),
            sub_identifier="grayscale",
            inputs=["LdrColor"],
            outputs=["LdrGrayscale"],
        ),
        SPGNode(
            "InvertKernel",
            os.path.join(spg_data_dir, "kernels", "InvertKernel.cu"),
            sub_identifier="invert",
            inputs=["Image"],
            outputs=["Inverted"],
            params={"strength": 1.0},
        ),
    ],
    connections=[
        ("LdrColor", "GrayscaleKernel.inputs:LdrColor"),
        ("GrayscaleKernel.outputs:LdrGrayscale", "InvertKernel.inputs:Image"),
        ("InvertKernel.outputs:Inverted", "LdrInverted"),
    ],
    render_product=render_product_path,
    copy_to=os.path.join(output_dir, "sensor_asset", "kernels"),
)
print(f"Authored grayscale+invert SPG on render product: {render_product_path}")

# Expose the final output AOV as a Replicator annotator so a writer can consume it
# (no viewport required), then drive it with the writer defined above.
rep.AnnotatorRegistry.register_annotator_from_aov(
    aov="LdrInverted", output_data_type=np.uint8, output_channels=4, is_gpu_enabled=True
)
rep.writers.register_writer(SingleAovWriter)
writer = rep.writers.get(
    "SingleAovWriter",
    init_params={"output_dir": output_dir, "aov_name": "LdrInverted"},
    render_products=[render_product_path],
    trigger=None,  # manual mode: write only the frame we schedule below
)

# Step the orchestrator synchronously (standalone workflow); several steps let the
# RTX renderer settle, then schedule a single write of the settled frame.
for _ in range(30):
    rep.orchestrator.step()
writer.schedule_write()
for _ in range(2):
    rep.orchestrator.step()
inverted = writer.last_frame
print(f"Saved inverted AOV under: {output_dir}")

if args.test:
    assert inverted.shape == (resolution[1], resolution[0], 4), f"unexpected AOV shape {inverted.shape}"
    assert int(inverted[..., :3].max()) > 0, "inverted AOV is all black"
    assert np.all(inverted[..., 0] == inverted[..., 1]) and np.all(
        inverted[..., 1] == inverted[..., 2]
    ), "inverted AOV is not neutral (R == G == B)"
    print("Grayscale+invert SPG example test checks passed.")

simulation_app.close()
