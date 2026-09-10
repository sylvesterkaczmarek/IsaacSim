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

"""Author an RTX Sensor Processing Graph (SPG) grayscale post-process.

This example demonstrates how to:

- Author a single-shader SPG with ``RtxCamera.author_spg`` and ``SPGNode`` from a
  user-provided CUDA kernel (``GrayscaleKernel.cu``) and its co-located Lua launch
  script (``GrayscaleKernel.cu.lua``).
- Optionally copy the kernel sources next to the sensor asset (``copy_to``).
- Read the custom output AOV (``LdrGrayscale``) back as a NumPy array via a
  Replicator annotator (``register_annotator_from_aov``) — no viewport required.

The kernel converts the rendered ``LdrColor`` AOV to grayscale using BT.601
luminance weights. See the SPG documentation for authoring details:
https://docs.omniverse.nvidia.com/kit/docs/omni.rtx.spg/0.2.0/Overview.html
"""

import argparse
import os

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="SPG grayscale post-process example.")
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


# Enable the SPG runtime (registers the rtx.spg.* plugins).
app_utils.enable_extension("omni.rtx.spg")
carb.settings.get_settings().set("/rtx/spg/enabled", True)

# Vendored SPG assets ship in this extension's data directory.
spg_data_dir = os.path.join(app_utils.get_extension_path("isaacsim.sensors.experimental.rtx"), "data", "spg")

output_dir = os.path.join(os.getcwd(), "_example_output_isaacsim.sensors.experimental.rtx", "spg_grayscale")
os.makedirs(output_dir, exist_ok=True)
resolution = (640, 360)  # (width, height)

# Load the Cornell box scene. Its camera already has the OmniSensorAPI schema, so
# RtxCamera can wrap it directly.
stage_utils.open_stage(usd_path=os.path.join(spg_data_dir, "scene", "cornell_box.usda"))
while stage_utils.is_stage_loading():
    simulation_app.update()

cam = RtxCamera("/World/Camera")

# A CameraSensor creates and drives the render product for the camera. Its
# ``resolution`` is (height, width).
sensor = CameraSensor(cam, resolution=(resolution[1], resolution[0]))
render_product_path = str(sensor.render_product.GetPath())

# Author the grayscale SPG onto the sensor's render product. ``copy_to`` collects
# the kernel + Lua launch script next to a would-be sensor asset; each shader's
# sourceAsset then points at the copied kernel.
cam.author_spg(
    SPGNode(
        "GrayscaleKernel",
        os.path.join(spg_data_dir, "kernels", "GrayscaleKernel.cu"),
        sub_identifier="grayscale",
        inputs=["LdrColor"],
        outputs=["LdrGrayscale"],
    ),
    connections=[
        ("LdrColor", "GrayscaleKernel.inputs:LdrColor"),
        ("GrayscaleKernel.outputs:LdrGrayscale", "LdrGrayscale"),
    ],
    render_product=render_product_path,
    copy_to=os.path.join(output_dir, "sensor_asset", "kernels"),
)
print(f"Authored grayscale SPG on render product: {render_product_path}")

# Expose the custom output AOV as a Replicator annotator so a writer can consume it
# (no viewport required), then drive it with the writer defined above.
rep.AnnotatorRegistry.register_annotator_from_aov(
    aov="LdrGrayscale", output_data_type=np.uint8, output_channels=4, is_gpu_enabled=True
)
rep.writers.register_writer(SingleAovWriter)
writer = rep.writers.get(
    "SingleAovWriter",
    init_params={"output_dir": output_dir, "aov_name": "LdrGrayscale"},
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
grayscale = writer.last_frame
print(f"Saved grayscale AOV under: {output_dir}")

if args.test:
    assert grayscale.shape == (resolution[1], resolution[0], 4), f"unexpected AOV shape {grayscale.shape}"
    assert int(grayscale[..., :3].max()) > 0, "grayscale AOV is all black"
    assert np.all(grayscale[..., 0] == grayscale[..., 1]) and np.all(
        grayscale[..., 1] == grayscale[..., 2]
    ), "grayscale AOV is not neutral (R == G == B)"
    print("Grayscale SPG example test checks passed.")

simulation_app.close()
