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

"""Subclass MobilityGenRobot with a custom (non-circular) footprint and
export occupancy-compatible trajectories.

The standard MobilityGen pipeline uses `occupancy_map_radius` to dilate the
occupancy map with a circular kernel.  For robots with rectangular or
irregular footprints, we override the buffering step so paths respect the
true swept volume.

Usage:
    from custom_footprint_robot import make_custom_footprint_robot
    RobotClass = make_custom_footprint_robot()
    # Then use RobotClass in MobilityGen's recording/replay pipeline.
"""

from __future__ import annotations

import numpy as np


def make_custom_footprint_robot():
    """Return a registered MobilityGenRobot subclass with a polygon footprint.

    Call at runtime when Isaac Sim is available.
    """
    import cv2
    from isaacsim.replicator.experimental.mobility_gen import ROBOTS, OccupancyMap
    from isaacsim.replicator.mobility_gen.examples.misc import HawkCamera
    from isaacsim.replicator.mobility_gen.examples.robots import WheeledMobilityGenRobot

    @ROBOTS.register()
    class CustomFootprintRobot(WheeledMobilityGenRobot):
        """Wheeled robot with a rectangular footprint for occupancy buffering.

        The footprint polygon (in meters, robot-centered) defines the true
        swept area used when dilating the occupancy map for path planning and
        collision checking.
        """

        physics_dt: float = 0.005
        z_offset: float = 0.25

        chase_camera_base_path = "chassis"
        chase_camera_x_offset: float = -2.0
        chase_camera_z_offset: float = 1.0
        chase_camera_tilt_angle: float = 60.0

        front_camera_base_path = "chassis/front_hawk"
        front_camera_rotation = (0.0, 0.0, 0.0)
        front_camera_translation = (0.2, 0.0, 0.1)
        front_camera_type = HawkCamera

        # Circular radii kept as conservative bounding values for any code
        # path that still reads these (e.g. A* heuristic bounds).
        occupancy_map_radius: float = 0.6
        occupancy_map_collision_radius: float = 0.55
        occupancy_map_z_min: float = 0.1
        occupancy_map_z_max: float = 0.6
        occupancy_map_cell_size: float = 0.05

        keyboard_linear_velocity_gain: float = 1.0
        keyboard_angular_velocity_gain: float = 1.0
        gamepad_linear_velocity_gain: float = 1.0
        gamepad_angular_velocity_gain: float = 1.0

        random_action_linear_velocity_range = (-0.3, 1.0)
        random_action_angular_velocity_range = (-0.75, 0.75)
        random_action_linear_acceleration_std: float = 5.0
        random_action_angular_acceleration_std: float = 5.0
        random_action_grid_pose_sampler_grid_size: float = 5.0
        path_following_speed: float = 0.8
        path_following_angular_gain: float = 1.0
        path_following_stop_distance_threshold: float = 0.5
        path_following_forward_angle_threshold = 0.785
        path_following_target_point_offset_meters: float = 1.0

        wheel_dof_names = ["left_wheel_joint", "right_wheel_joint"]
        usd_url: str = "/path/to/custom_robot.usd"
        chassis_subpath: str = "chassis"
        wheel_base: float = 0.5
        wheel_radius: float = 0.1

        # --- Custom footprint (meters, robot-centered) ---
        # Rectangle 1.0m long x 0.5m wide (front-heavy)
        footprint_polygon_meters: list[tuple[float, float]] = [
            (0.6, 0.25),  # front-right
            (0.6, -0.25),  # front-left
            (-0.4, -0.25),  # rear-left
            (-0.4, 0.25),  # rear-right
        ]

        @classmethod
        def footprint_kernel(cls, resolution: float) -> np.ndarray:
            """Build a binary dilation kernel from the footprint polygon.

            Args:
                resolution: Map resolution in meters/pixel.

            Returns:
                Binary uint8 kernel suitable for cv2.dilate.
            """
            pts_pixels = np.array(
                [(x / resolution, y / resolution) for x, y in cls.footprint_polygon_meters],
                dtype=np.float32,
            )
            # Center the polygon in a tight bounding kernel
            mins = pts_pixels.min(axis=0)
            maxs = pts_pixels.max(axis=0)
            size = (maxs - mins).astype(int) + 1
            # Shift so all coords are positive pixel indices
            shifted = (pts_pixels - mins).astype(np.int32)
            kernel = np.zeros((size[1], size[0]), dtype=np.uint8)
            cv2.fillConvexPoly(kernel, shifted.reshape(-1, 1, 2), 255)
            return kernel

        @classmethod
        def buffer_occupancy_map(cls, occupancy_map: OccupancyMap) -> OccupancyMap:
            """Dilate the occupancy map using the robot's polygon footprint.

            Replaces the default circular `buffered_meters()` call for path
            planning and spawn-point validation.

            Args:
                occupancy_map: The raw occupancy map.

            Returns:
                A new OccupancyMap with occupied regions dilated by the
                footprint polygon.
            """
            # Not re-exported at package level, so import from the impl module.
            from isaacsim.replicator.experimental.mobility_gen.impl.occupancy_map import OccupancyMapDataValue

            kernel = cls.footprint_kernel(occupancy_map.resolution)
            occupied = (occupancy_map.occupied_mask()).astype(np.uint8) * 255
            dilated = cv2.dilate(occupied, kernel, iterations=1)
            new_data = occupancy_map.data.copy()
            new_data[dilated > 0] = OccupancyMapDataValue.OCCUPIED
            return OccupancyMap(
                data=new_data,
                resolution=occupancy_map.resolution,
                origin=occupancy_map.origin,
            )

    return CustomFootprintRobot


# ---------------------------------------------------------------------------
# Recording with polygon-buffered occupancy map
# ---------------------------------------------------------------------------


def record_with_custom_footprint(
    scene_usd: str,
    omap_yaml: str,
    num_episodes: int = 5,
    max_steps: int = 2000,
    output_dir: str = None,
) -> None:
    """Record trajectories using the polygon footprint for map dilation.

    After scenario construction, replaces the circular buffered map with one
    dilated by the true robot polygon, so planned paths respect the actual
    swept volume.

    Args:
        scene_usd: Path to the environment USD file.
        omap_yaml: Path to the ROS-format occupancy map YAML.
        num_episodes: Number of episodes to record.
        max_steps: Maximum physics steps per episode.
        output_dir: Output directory; defaults to ~/MobilityGenData/recordings.
    """
    import os
    import tempfile

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(launch_config={"headless": True, "multi_gpu": False})

    import omni.timeline
    from isaacsim.core.experimental.utils.stage import open_stage, save_stage
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.replicator.experimental.mobility_gen import SCENARIOS, OccupancyMap, RecordingSession

    robot_cls = make_custom_footprint_robot()
    scenario_cls = SCENARIOS.get("RandomPathFollowingScenario")

    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "MobilityGenData", "recordings")
    os.makedirs(output_dir, exist_ok=True)

    occupancy_map = OccupancyMap.from_ros_yaml(omap_yaml)

    opened, _ = open_stage(scene_usd)
    if not opened:
        simulation_app.close()
        raise RuntimeError(f"Could not open scene USD: {scene_usd}")
    simulation_app.update()

    cached_stage = os.path.join(tempfile.mkdtemp(), "stage.usd")
    save_stage(cached_stage)

    # RecordingSession owns the ground plane, robot spawn, Config and writer.
    session = RecordingSession()
    scenario_instance = session.build(
        robot_cls,
        scenario_cls,
        occupancy_map,
        scene_usd=scene_usd,
        cached_stage_path=cached_stage,
        recordings_dir=output_dir,
    )

    # Replace the circular buffer with the polygon-dilated map
    scenario_instance.buffered_occupancy_map = robot_cls.buffer_occupancy_map(occupancy_map)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    session.initialize()

    for episode in range(num_episodes):
        session.reset()
        session.enable_recording()

        for _ in range(max_steps):
            # initialize_physics() does not start the Kit timeline, so physics
            # needs an explicit step; simulation_app.update() alone will not tick it.
            SimulationManager.step(steps=1)
            simulation_app.update()
            if not session.step(robot_cls.physics_dt):
                break

        print(f"Episode {episode + 1}/{num_episodes}: {session.step_count} steps -> {session.recording_path}")
        session.disable_recording()

    session.close()
    omni.timeline.get_timeline_interface().stop()
    simulation_app.close()


# ---------------------------------------------------------------------------
# Trajectory export utility
# ---------------------------------------------------------------------------


def export_trajectory_for_occupancy(
    recording_dir: str,
    output_path: str,
) -> None:
    """Load a MobilityGen recording and export an occupancy-compatible trajectory.

    Writes a single .npz file containing:
        - positions: (N, 3) world-frame positions
        - orientations: (N, 4) quaternions (w, x, y, z)
        - timestamps: (N,) time in seconds from episode start
        - footprint_polygon: (M, 2) the robot's footprint vertices in meters
        - occupancy_map_resolution: scalar
        - occupancy_map_origin: (3,) [x, y, yaw]

    This format is directly consumable by occupancy-map-based planners that
    need to verify trajectory feasibility against a known map.

    Args:
        recording_dir: Path to a MobilityGen recording directory (contains
            state/common/*.npz and occupancy_map/map.yaml).
        output_path: Destination .npz file path.
    """
    import glob
    import os

    import yaml

    # Load occupancy map metadata
    map_yaml_path = os.path.join(recording_dir, "occupancy_map", "map.yaml")
    with open(map_yaml_path) as f:
        map_meta = yaml.safe_load(f)

    # Load per-step state (keys are prefixed: "robot.position", "robot.orientation")
    state_dir = os.path.join(recording_dir, "state", "common")
    step_files = sorted(glob.glob(os.path.join(state_dir, "*.npz")))

    positions = []
    orientations = []
    for step_file in step_files:
        data = np.load(step_file)
        positions.append(data["robot.position"])
        orientations.append(data["robot.orientation"])

    robot_cls = make_custom_footprint_robot()
    physics_dt = robot_cls.physics_dt

    positions = np.array(positions)
    orientations = np.array(orientations)
    timestamps = np.arange(len(positions)) * physics_dt

    np.savez(
        output_path,
        positions=positions,
        orientations=orientations,
        timestamps=timestamps,
        footprint_polygon=np.array(robot_cls.footprint_polygon_meters),
        occupancy_map_resolution=map_meta["resolution"],
        occupancy_map_origin=np.array(map_meta["origin"]),
    )
