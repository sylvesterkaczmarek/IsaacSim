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
"""Draw an RTX Lidar point cloud colored by a per-point scalar field (distance, intensity, ...).

The whole OmniGraph (IsaacExtractRTXSensorPointCloud -> IsaacMapScalarsToColors ->
DebugDrawPointCloud, with a frame-sync gate) is built by a single helper:
``isaacsim.sensors.rtx.nodes.register_scalar_colored_point_cloud_writer``. This example just picks
the scalar to color by.
"""

import argparse

from isaacsim import SimulationApp

TEST_NUM_FRAMES = 120

parser = argparse.ArgumentParser(description="Draw an RTX Lidar point cloud colored by a scalar field.")
parser.add_argument("-c", "--config", default="Example_Rotary", type=str, help="Lidar config name.")
parser.add_argument(
    "--scalar", default="distance", type=str, help="Scalar field to color by (e.g. distance, intensity)."
)
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode.")
parser.add_argument("--headless", default=False, action="store_true", help="Run in headless mode.")
parser.add_argument(
    "--max-frames", default=0, type=int, help="Maximum number of simulation frames. Use 0 to run until closed."
)
parser.add_argument("--size", default=0.05, type=float, help="Size of the drawn points in world units.")
parser.add_argument(
    "--base-color",
    default=[1.0, 1.0, 1.0, 1.0],
    nargs=4,
    type=float,
    metavar=("R", "G", "B", "A"),
    help="RGBA multiplier applied to the color ramp.",
)
scale_group = parser.add_mutually_exclusive_group()
scale_group.add_argument("--log-scale", dest="log_scale", action="store_true", help="Use log scaling.")
scale_group.add_argument("--linear-scale", dest="log_scale", action="store_false", help="Use linear scaling (default).")
parser.set_defaults(log_scale=False)
args, _ = parser.parse_known_args()

# Launch the app before importing simulation-dependent modules.
simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni.usd
from isaacsim.core.experimental.objects import Cube, DistantLight
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor

app_utils.enable_extension("isaacsim.core.nodes")
app_utils.enable_extension("isaacsim.sensors.rtx.nodes")
app_utils.enable_extension("isaacsim.util.debug_draw")

from isaacsim.sensors.rtx.nodes import register_scalar_colored_point_cloud_writer


def create_distance_scene() -> None:
    """Create cube targets at clear horizontal offsets and varied ranges for the lidar to scan."""
    omni.usd.get_context().new_stage()

    light = DistantLight("/World/light")
    light.set_intensities(5000.0)

    cube_scale = np.array([0.7, 0.7, 0.7])
    cube_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    cube_positions = [
        np.array([2.0, -2.0, 1.0]),
        np.array([3.5, -1.0, 1.0]),
        np.array([5.0, 0.0, 1.0]),
        np.array([6.5, 1.0, 1.0]),
        np.array([8.0, 2.0, 1.0]),
    ]
    for index, position in enumerate(cube_positions):
        Cube(f"/World/distance_cube_{index:02d}", positions=position, orientations=cube_orientation, scales=cube_scale)


# Build the extract -> map-scalars -> debug-draw graph for the chosen scalar and get its writer name.
writer_name = register_scalar_colored_point_cloud_writer(
    scalar=args.scalar, base_color=args.base_color, log_scale=args.log_scale
)
create_distance_scene()

lidar = Lidar.create(
    "/World/lidar",
    config=args.config,
    translations=np.array([0.0, 0.0, 0.75]),
    aux_output_level="FULL",
)

sensor = LidarSensor(lidar, annotators=[])
writer = sensor.attach_writer(writer_name, size=args.size)

print(f"Drawing RTX Lidar points colored by {args.scalar} ({writer_name}, {writer.__class__.__name__}).")

app_utils.play()

frame_count = 0
frame_limit = args.max_frames
if args.test and frame_limit <= 0:
    frame_limit = TEST_NUM_FRAMES

while simulation_app.is_running():
    simulation_app.update()
    frame_count += 1
    if frame_limit > 0 and frame_count >= frame_limit:
        break

app_utils.stop()
simulation_app.close()
