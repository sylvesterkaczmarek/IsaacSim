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

"""Author a depth-sensor USD asset, then drive it by attaching to its embedded render product.

Exports an ``RtxCamera`` plus a template ``RenderProduct`` (``OmniSensorDepthSensorSingleViewAPI``
schema, resolution, depth render vars), then loads it back and wraps the camera with
``SingleViewDepthCameraSensor``, which attaches directly to that render product -- deriving
resolution and annotators from the asset -- instead of creating a new one.
"""

import argparse
import os

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Create and drive a depth camera sensor asset.")
parser.add_argument("--test", default=False, action="store_true", help="Run headless and exit after a few frames.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.test})

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.timeline
from isaacsim.core.experimental.objects import Cone, Cube
from isaacsim.core.experimental.utils.transform import euler_angles_to_quaternion
from isaacsim.sensors.experimental.rtx import RtxCamera, SingleViewDepthCameraSensor
from isaacsim.storage.native.nucleus import get_assets_root_path
from pxr import Gf, UsdRender

output_dir = os.path.join(
    os.getcwd(), "_example_output_isaacsim.sensors.experimental.rtx", "create_camera_depth_sensor"
)
os.makedirs(output_dir, exist_ok=True)

# Author the asset: RtxCamera + template RenderProduct (schema + baseline + resolution + render vars).
root = stage_utils.define_prim("/root", "Xform")
RtxCamera("/root/Camera")
stage_utils.define_prim("/root/TemplateRenderProduct", "Scope")
rp = UsdRender.Product(
    SingleViewDepthCameraSensor.add_template_render_product(
        parent_prim_path="/root/TemplateRenderProduct",
        camera_prim_path="/root/Camera",
        **{"omni:rtx:post:depthSensor:baselineMM": 42.0},
    )
)
rp.CreateResolutionAttr(Gf.Vec2i(640, 480))  # (width, height)
stage = stage_utils.get_current_stage(backend="usd")
ordered = []
for name in ("DepthSensorDistance", "DepthSensorImager", "DepthSensorPointCloudColor", "DepthSensorPointCloudPosition"):
    var = UsdRender.Var.Define(stage, f"/root/TemplateRenderProduct/{name}")
    var.CreateSourceNameAttr(name)
    ordered.append(var.GetPath())
rp.CreateOrderedVarsRel().SetTargets(ordered)
stage.SetDefaultPrim(root)
asset_path = os.path.join(output_dir, "example_camera_with_depth_sensor.usda")
stage.Export(asset_path)

# Load the asset into a fresh stage with the standard depth scene + camera pose.
stage_utils.create_new_stage()
stage_utils.add_reference_to_stage(
    usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/gridroom_black.usd", path="/World/black_grid"
)
Cube("/cube_1", sizes=1.0, positions=np.array([0.25, 0.25, 0.25]), scales=np.array([0.5, 0.5, 0.5]))
Cube("/cube_2", sizes=1.0, positions=np.array([-1.0, -1.0, 0.25]))
Cone("/cone", radii=0.5, heights=1.0, positions=np.array([-0.1, -0.3, 0.2]))

# Reference the .usda asset (Xform root) and wrap its Camera directly; with no resolution/annotators
# passed, SingleViewDepthCameraSensor attaches to the embedded render product (derived from the asset).
stage_utils.add_reference_to_stage(usd_path=asset_path, path="/World/DepthCamera", prim_type="Xform")
camera = RtxCamera(
    "/World/DepthCamera/Camera",
    translations=np.array([3.0, 0.0, 0.6]),
    orientations=euler_angles_to_quaternion(np.array([90, 90, 0]), degrees=True, extrinsic=False).numpy(),
)
sensor = SingleViewDepthCameraSensor(camera)
sensor.set_enabled_post_processing(True)
print(
    f"Attached to {sensor.render_product.GetPath()} | resolution {sensor.resolution} | "
    f"annotators {sensor.annotators} | baseline {sensor.get_sensor_baseline()} mm"
)

# Run until depth is available.
omni.timeline.get_timeline_interface().play()
frame = 0
while simulation_app.is_running():
    simulation_app.update()
    frame += 1
    depth, _ = sensor.get_data("depth_sensor_distance")
    if depth is not None:
        d = depth.numpy()
        print(f"Frame {frame}: depth_sensor_distance {d.shape}, range [{d.min():.2f}, {d.max():.2f}] m")
        break
    if args.test and frame >= 30:
        break

simulation_app.close()
