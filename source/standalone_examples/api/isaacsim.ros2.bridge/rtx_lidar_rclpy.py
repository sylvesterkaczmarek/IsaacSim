# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Publish an RTX lidar point cloud with rclpy, without the OmniGraph publisher nodes.

The lidar's GenericModelOutput arrives as separate per-field host arrays (x, y, z,
intensity). ``fill_point_cloud2_message`` from ``isaacsim.ros2.nodes`` packs them into
the ``sensor_msgs/PointCloud2`` ``point_step`` layout with the same parallel C++
interleave the OmniGraph publisher uses, reusing the message's data storage across
frames. Inspect the output with: ``ros2 topic echo /point_cloud --no-arr``
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument("--headless", default=False, action="store_true", help="Run in headless mode.")
parser.add_argument(
    "--frames", type=int, default=0, help="Number of point clouds to publish before exiting (0 = run until closed)."
)
parser.add_argument(
    "--profile",
    action="store_true",
    help="Enable the Tracy profiler backend; attach the Tracy GUI or capture tool to port 8086.",
)
args, _ = parser.parse_known_args()

# Stream the carb.profiler zones (the ``rclpy:*`` zones below, plus Isaac's own) to Tracy when
# profiling. This must be set in the SimulationApp config; it is not a runtime setting.
config = {"headless": args.headless}
if args.profile:
    config["profiler_backend"] = ["tracy"]
simulation_app = SimulationApp(config)

import contextlib

import carb.profiler
import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni.usd
from isaacsim.core.experimental.objects import Cube, DomeLight
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data

# Enable the ROS 2 bridge to set up the bundled rclpy environment.
if args.profile:
    app_utils.enable_extension("omni.kit.profiler.tracy")
app_utils.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()


@contextlib.contextmanager
def _profile(name: str):
    """Emit a ``carb.profiler`` zone (visible in Tracy / the profiler window).

    A no-op unless a profiler backend is active, so it is free when the example is run normally.
    """
    carb.profiler.begin(0, name)
    try:
        yield
    finally:
        carb.profiler.end(0)


# rclpy resolves through the configured ROS 2 environment: a sourced system installation is
# used when its Python matches Kit's (e.g. Jazzy on Ubuntu 24.04); with nothing sourced, the
# libraries bundled with the bridge are used.
import rclpy
from isaacsim.ros2.nodes import fill_point_cloud2_message
from sensor_msgs.msg import PointCloud2

# Start from a fresh stage with a light and nearby cubes so the lidar has geometry to scan.
omni.usd.get_context().new_stage()
DomeLight("/World/DomeLight").set_intensities(500)
Cube("/World/cube_front", positions=np.array([5.0, 0.0, 0.0]), scales=np.array([2.0, 2.0, 2.0]))
Cube("/World/cube_left", positions=np.array([0.0, 5.0, 0.0]), scales=np.array([2.0, 2.0, 2.0]))
Cube("/World/cube_right", positions=np.array([0.0, -5.0, 0.0]), scales=np.array([2.0, 2.0, 2.0]))

# Create a Cartesian, world-frame lidar so the published points are directly usable.
lidar = Lidar.create(
    "/World/lidar",
    config="Example_Rotary",
    translations=np.array([0.0, 0.0, 1.0]),
    attributes={
        "omni:sensor:Core:elementsCoordsType": "CARTESIAN",
        "omni:sensor:Core:outputFrameOfReference": "WORLD",
    },
)

# The sensor wrapper attaches the GenericModelOutput annotator; get_data() polls it each frame.
sensor = LidarSensor(lidar, annotators=["generic-model-output"])

# Create the rclpy publisher and one PointCloud2 message that is refilled every frame.
rclpy.init()
node = rclpy.create_node("isaac_rtx_lidar_publisher")
publisher = node.create_publisher(PointCloud2, "point_cloud", 10)
message = PointCloud2()
message.header.frame_id = "world"

app_utils.play()

max_frames = args.frames if args.frames > 0 else (100 if args.test else 0)
published = 0
updates = 0
last_scan_timestamp = 0
while simulation_app.is_running():
    simulation_app.update()
    updates += 1
    if args.test and updates >= 2000:
        print(f"Stopping after {updates} updates with {published} point clouds published.")
        break

    data, _ = sensor.get_data("generic-model-output")
    if data is None:
        continue
    with _profile("rclpy:parse_gmo"):
        gmo = parse_generic_model_output_data(data)
    # get_data() returns the latest scan, so consecutive frames can see the same one;
    # publish each scan once, keyed by its authoritative scan-start time.
    if gmo.numElements == 0 or int(gmo.timestampNs) == last_scan_timestamp:
        continue
    last_scan_timestamp = int(gmo.timestampNs)

    # Gather the separate per-field host arrays into the message's packed layout.
    with _profile("rclpy:gather_xyz"):
        xyz = np.column_stack((gmo.x, gmo.y, gmo.z))
        intensity = np.ascontiguousarray(gmo.scalar, dtype=np.float32)
    message.header.stamp = node.get_clock().now().to_msg()
    with _profile("rclpy:fill_message"):
        fill_point_cloud2_message(message, xyz, [("intensity", intensity)])
    with _profile("rclpy:publish"):
        publisher.publish(message)

    published += 1
    if published == 1 or published % 30 == 0:
        print(f"Published point cloud {published}: {message.width} points, {message.point_step} bytes/point")
    if max_frames > 0 and published >= max_frames:
        break

if args.test and published == 0:
    raise RuntimeError("test mode: no point cloud was published (no valid lidar data arrived)")

node.destroy_node()
rclpy.shutdown()
simulation_app.close()
