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
"""Import a point cloud into the viewport."""

import argparse
import os

TEST_NUM_FRAMES = 100

parser = argparse.ArgumentParser(description="Import a point cloud into the viewport.")
parser.add_argument("--input-file", type=str, required=True, help="Path to the .npz point cloud file to import.")
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode.")
parser.add_argument("--size", default=0.01, type=float, help="Size of the points in the viewport.")
parser.add_argument(
    "--color",
    default=[0.0, 1.0, 0.5, 1.0],
    nargs=4,
    type=float,
    metavar=("R", "G", "B", "A"),
    help="RGBA color of the points (values 0-1).",
)
args, _ = parser.parse_known_args()

input_file = os.path.abspath(args.input_file)
if not os.path.isfile(input_file):
    parser.error(f"Point cloud file does not exist: {input_file}")

from isaacsim import SimulationApp

# Launch the app with rendering enabled so the point cloud can be viewed.
simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni.graph.core as og

# Load point data from disk.
with np.load(input_file) as cloud:
    data = np.ascontiguousarray(cloud["points"], dtype=np.float32)

try:
    # Build an action graph that sends the point buffer to the debug draw node each tick.
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("DebugDrawPointCloud", "isaacsim.util.debug_draw.DebugDrawPointCloud"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "DebugDrawPointCloud.inputs:exec"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("DebugDrawPointCloud.inputs:dataPtr", data.ctypes.data),
                ("DebugDrawPointCloud.inputs:bufferSize", data.nbytes),
                ("DebugDrawPointCloud.inputs:size", args.size),
                ("DebugDrawPointCloud.inputs:color", args.color),
            ],
        },
    )
except Exception as e:
    print(e)

# Run the simulation so the debug draw node continues to render the cloud.
app_utils.play()
frame_count = 0
while simulation_app.is_running():
    if args.test and frame_count >= TEST_NUM_FRAMES:
        break

    simulation_app.update()
    frame_count += 1

# Stop playback and close the app.
app_utils.stop()
simulation_app.close()
