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
"""Draw an RTX Radar point cloud colored by per-point radial velocity (Doppler).

Radial velocity is a radar-only quantity (it comes from the radar auxiliary ``rv_ms`` field),
so this example uses an RTX Radar rather than a lidar. It reuses the same coloring graph as
``test_lidar_point_cloud_coloring.py`` -- ``register_scalar_colored_point_cloud_writer`` builds
IsaacExtractRTXSensorPointCloud -> IsaacMapScalarsToColors -> DebugDrawPointCloud -- just with
``scalar="radialVelocityMS"``.

Two radar specifics:
  * Motion BVH must be enabled (``enable_motion_bvh``) for the radar to estimate Doppler.
  * Radial velocity is ~0 for a static scene, so the target cubes are oscillated along the
    radar line-of-sight each frame; RTX derives the per-point velocity from the resulting
    motion, and the points animate between the low (blue) and high (red) ends of the ramp.
"""

import argparse

from isaacsim import SimulationApp

TEST_NUM_FRAMES = 240

parser = argparse.ArgumentParser(description="Draw an RTX Radar point cloud colored by radial velocity.")
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode.")
parser.add_argument("--headless", default=False, action="store_true", help="Run in headless mode.")
parser.add_argument(
    "--max-frames", default=0, type=int, help="Maximum number of simulation frames. Use 0 to run until closed."
)
parser.add_argument("--size", default=0.2, type=float, help="Size of the drawn points in world units.")
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

# Launch the app before importing simulation-dependent modules. Motion BVH is required for radar Doppler.
simulation_app = SimulationApp({"headless": args.headless, "enable_motion_bvh": True})

import carb
import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni.usd
from isaacsim.core.experimental.objects import Cube, DistantLight
from isaacsim.sensors.experimental.rtx import Radar, RadarSensor

# Belt-and-suspenders: make sure the ray-traced motion BVH the radar reads from is on.
settings = carb.settings.get_settings()
settings.set("/renderer/raytracingMotion/enabled", True)
settings.set("/renderer/raytracingMotion/enableHydraEngineMasking", True)
settings.set("/renderer/raytracingMotion/enabledForHydraEngines", "0,1,2,3,4")

app_utils.enable_extension("isaacsim.core.nodes")
app_utils.enable_extension("isaacsim.sensors.rtx.nodes")
app_utils.enable_extension("isaacsim.util.debug_draw")

from isaacsim.sensors.rtx.nodes import register_scalar_colored_point_cloud_writer

# Target cubes spread across the radar's forward (+X) field of view, at varied ranges.
TARGET_BASE_POSITIONS = np.array(
    [
        [3.0, -1.5, 1.0],
        [4.0, -0.75, 1.0],
        [5.0, 0.0, 1.0],
        [6.0, 0.75, 1.0],
        [7.0, 1.5, 1.0],
    ]
)
# Each cube oscillates along X (toward/away from the radar) so it has a nonzero radial velocity.
OSCILLATION_AMPLITUDE = 0.8  # meters
OSCILLATION_PHASE_STEP = 0.06  # radians per frame


def create_radial_velocity_scene() -> Cube:
    """Create the oscillating target cubes (plus a light) and return the Cube wrapper for all of them.

    Returns:
        Batched cube wrapper for all radar targets.
    """
    omni.usd.get_context().new_stage()

    light = DistantLight("/World/light")
    light.set_intensities(5000.0)

    paths = [f"/World/target_{index:02d}" for index in range(len(TARGET_BASE_POSITIONS))]
    return Cube(paths=paths, positions=TARGET_BASE_POSITIONS, sizes=0.7)


# Build the extract -> map-scalars -> debug-draw graph for radial velocity and get its writer name.
writer_name = register_scalar_colored_point_cloud_writer(
    scalar="radialVelocityMS", base_color=args.base_color, log_scale=args.log_scale
)
targets = create_radial_velocity_scene()

# A static radar at the origin; with a static sensor, the default (sensor) frame already reports the
# targets' world motion as radial velocity. aux_output_level="BASIC" is what populates rv_ms.
radar = Radar(
    "/World/radar",
    translations=np.array([0.0, 0.0, 1.0]),
    orientations=np.array([1.0, 0.0, 0.0, 0.0]),
    aux_output_level="BASIC",
)

sensor = RadarSensor(radar, annotators=[])
writer = sensor.attach_writer(writer_name, size=args.size)

print(f"Drawing RTX Radar points colored by radial velocity ({writer_name}, {writer.__class__.__name__}).")

app_utils.play()

# Pre-compute the per-cube oscillation phases so the targets move out of sync (a spread of velocities).
phases = np.linspace(0.0, 2.0 * np.pi, len(TARGET_BASE_POSITIONS), endpoint=False)
base_x = TARGET_BASE_POSITIONS[:, 0]

frame_count = 0
frame_limit = args.max_frames
if args.test and frame_limit <= 0:
    frame_limit = TEST_NUM_FRAMES

while simulation_app.is_running():
    # Oscillate the cubes along the line-of-sight before stepping so RTX picks up the motion this frame.
    positions = TARGET_BASE_POSITIONS.copy()
    positions[:, 0] = base_x + OSCILLATION_AMPLITUDE * np.sin(OSCILLATION_PHASE_STEP * frame_count + phases)
    targets.set_world_poses(positions=positions)

    simulation_app.update()
    frame_count += 1
    if frame_limit > 0 and frame_count >= frame_limit:
        break

app_utils.stop()
simulation_app.close()
