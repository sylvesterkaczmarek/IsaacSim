# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Export a point cloud from a Lidar sensor to a file."""

import argparse
import os
from typing import Any

from isaacsim import SimulationApp

output_dir = os.path.join(os.getcwd(), "_example_output_isaacsim.sensors.experimental.rtx", "export_point_cloud")

parser = argparse.ArgumentParser(description="Export a point cloud from a Lidar sensor to a file.")
parser.add_argument(
    "--output-file",
    type=str,
    default=os.path.join(output_dir, "point_cloud.npz"),
    help="File to write the point cloud to.",
)
parser.add_argument(
    "--max-frames",
    type=int,
    default=0,
    help="Optional maximum number of simulation frames to wait for a valid point cloud. Use 0 to wait until one arrives.",
)
parser.add_argument("--headless", default=False, action="store_true", help="Run in headless mode.")
args, _ = parser.parse_known_args()

# Launch the app before importing simulation-dependent modules.
simulation_app = SimulationApp({"headless": args.headless})

# Create the example output directory used by the other sensor examples.
os.makedirs(output_dir, exist_ok=True)

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.experimental.objects import Cube, DomeLight
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data
from omni.replicator.core import Writer

# Start from a fresh stage for the lidar scene.
omni.usd.get_context().new_stage()

# Resolve the point cloud path and create its parent directory for custom overrides.
output_file = os.path.abspath(args.output_file)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Add a light and nearby cubes so the lidar has geometry to scan.
dome_light = DomeLight("/World/DomeLight")
dome_light.set_intensities(500)

Cube("/World/cube_front", positions=np.array([5.0, 0.0, 0.0]), scales=np.array([2.0, 2.0, 2.0]))
Cube("/World/cube_left", positions=np.array([0.0, 5.0, 0.0]), scales=np.array([2.0, 2.0, 2.0]))
Cube("/World/cube_right", positions=np.array([0.0, -5.0, 0.0]), scales=np.array([2.0, 2.0, 2.0]))

# Create a Cartesian, world-frame lidar so the saved data is directly reusable.
lidar = Lidar.create(
    "/World/lidar",
    config="Example_Rotary",
    translations=np.array([0.0, 0.0, 1.0]),
    attributes={
        "omni:sensor:Core:elementsCoordsType": "CARTESIAN",
        "omni:sensor:Core:outputFrameOfReference": "WORLD",
    },
)

# The custom writer brings its own GenericModelOutput annotator.
sensor = LidarSensor(lidar, annotators=[])


class LidarPointCloudDiskWriter(Writer):
    """Write the first valid Cartesian Lidar point cloud to disk.

    Args:
        output_file: Destination for the compressed point-cloud archive.
    """

    def __init__(self, output_file: str = "point_cloud.npz") -> None:
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("GenericModelOutput")]
        self.output_file = os.path.abspath(output_file)
        self.capture_count = 0

    def write(self, data: dict[str, Any]) -> None:
        """Write the first valid point cloud in a render-product payload.

        Args:
            data: Writer payload containing render-product GMO data.
        """
        # Save only the first valid point cloud.
        if self.capture_count > 0 or "renderProducts" not in data:
            return

        for _rp_name, rp_data in data["renderProducts"].items():
            # Parse the raw GenericModelOutput buffer from the render product.
            gmo_raw = rp_data.get("GenericModelOutput")
            if isinstance(gmo_raw, dict):
                gmo_raw = gmo_raw.get("data")
            gmo = parse_generic_model_output_data(gmo_raw)
            if gmo.numElements > 0:
                # Store the XYZ coordinates as a contiguous Nx3 float32 array.
                points = np.ascontiguousarray(np.column_stack((gmo.x, gmo.y, gmo.z)), dtype=np.float32)
                np.savez(
                    self.output_file,
                    points=points,
                )
                self.capture_count += 1
                print(f"Saved {points.shape[0]} points to {self.output_file}")
                break


# Register and attach the disk writer to the lidar sensor.
rep.WriterRegistry.register(LidarPointCloudDiskWriter)
writer = sensor.attach_writer("LidarPointCloudDiskWriter", output_file=output_file)

# Run the simulation until a cloud is captured or the optional frame limit is reached.
app_utils.play()

frames_waited = 0
while simulation_app.is_running():
    if writer.capture_count:
        break

    if args.max_frames > 0 and frames_waited >= args.max_frames:
        print(f"No valid point cloud received after {frames_waited} simulation frames.")
        break

    simulation_app.update()
    frames_waited += 1

# Stop playback and close the app.
app_utils.stop()
simulation_app.close()
