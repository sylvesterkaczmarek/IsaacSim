# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Provides shared fixtures, tolerances, and scene helpers for physics sensor OmniGraph node tests. Covers timeline reset helpers, gravity constants, and reusable ant and cube scene configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path_async
from pxr import PhysxSchema, Usd, UsdPhysics

EARTH_GRAVITY = 9.81
MOON_GRAVITY = 1.62
CM_GRAVITY = 981.0

GRAVITY_TOLERANCE = 0.1
ANGLE_TOLERANCE_DEG = 0.1
ANGULAR_VEL_TOLERANCE = 0.2
ORIENTATION_TOLERANCE = 1e-4
SMALL_TOLERANCE = 0.01


async def step_simulation(seconds: float) -> None:
    """Advance Kit updates for the requested duration using the configured physics timestep.

    Args:
        seconds: Duration to advance.
    """
    dt = SimulationManager.get_physics_dt()
    steps = max(1, int(round(seconds / dt)))
    for _ in range(steps):
        await omni.kit.app.get_app().next_update_async()


async def reset_timeline(timeline: Any = None, *, steps: int = 2) -> None:
    """Stop and restart the timeline, then advance a few frames to refresh sensor outputs.

    Args:
        timeline: Timeline interface to reset.
        steps: Number of updates to run after restarting.
    """
    if timeline is None:
        timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()
    timeline.play()
    for _ in range(steps):
        await omni.kit.app.get_app().next_update_async()


@dataclass
class AntConfig:
    """Configuration data for ant robot used in sensor tests."""

    leg_paths: list[str] = field(default_factory=lambda: [f"/Ant/Arm_{i + 1:02d}/Lower_Arm" for i in range(4)])
    sphere_path: str = "/Ant/Sphere"
    sensor_offsets: list[np.ndarray] = field(default_factory=lambda: [np.array([[40.0, 0.0, 0.0]]) for _ in range(4)])
    # IMU sensor offsets (at origin for each sensor location)
    imu_sensor_offsets: list[np.ndarray] = field(
        default_factory=lambda: [np.array([[0.0, 0.0, 0.0]]) for _ in range(5)]
    )
    # IMU sensor orientations (identity quaternions, wxyz)
    sensor_quatd: list[np.ndarray] = field(default_factory=lambda: [np.array([[1.0, 0.0, 0.0, 0.0]]) for _ in range(5)])
    colors: list[tuple[float, float, float, float]] = field(
        default_factory=lambda: [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
    )
    shoulder_joints: list[str] = field(
        default_factory=lambda: [f"/Ant/Arm_{i + 1:02d}/Upper_Arm/shoulder_joint" for i in range(4)]
    )
    lower_joints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Derive lower-arm joint paths from the configured Ant lower-arm link paths."""
        if not self.lower_joints:
            self.lower_joints = [f"{path}/elbow_joint" for path in self.leg_paths]


def author_ant_standing_drives(config: AntConfig, stage: Usd.Stage) -> None:
    """Author drives that bend the Ant legs into their standing pose.

    The legacy Ant is centimeter-scaled, so retain its stiffness of 100 while
    using the 10:1 stiffness-to-damping ratio from the experimental sensor
    tests. Its authored damping of 100 prevents it from reaching the standing
    targets before falling.

    Args:
        config: Ant prim paths used by the test scene.
        stage: Stage containing the Ant articulation.
    """
    for shoulder_path in config.shoulder_joints:
        shoulder = stage.GetPrimAtPath(shoulder_path)
        rot_x_drive = UsdPhysics.DriveAPI.Apply(shoulder, "rotX")
        rot_x_drive.CreateTargetPositionAttr().Set(30.0)
        rot_x_drive.CreateStiffnessAttr().Set(100.0)
        rot_x_drive.CreateDampingAttr().Set(10.0)

        rot_z_drive = UsdPhysics.DriveAPI.Apply(shoulder, "rotZ")
        rot_z_drive.CreateTargetPositionAttr().Set(0.0)
        rot_z_drive.CreateStiffnessAttr().Set(100.0)
        rot_z_drive.CreateDampingAttr().Set(10.0)

    for elbow_path in config.lower_joints:
        elbow = stage.GetPrimAtPath(elbow_path)
        drive = UsdPhysics.DriveAPI.Apply(elbow, "angular")
        drive.CreateTargetPositionAttr().Set(90.0)
        drive.CreateStiffnessAttr().Set(100.0)
        drive.CreateDampingAttr().Set(10.0)


def disable_articulation_sleep(robot_path: str = "/Ant", stage: Usd.Stage | None = None) -> None:
    """Disable PhysX sleep/stabilization under the robot so resting contacts keep reporting.

    Args:
        robot_path: USD path to the ant robot root prim.
        stage: Stage to modify. Uses the USD context stage when ``None``.
    """
    if stage is None:
        stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        return
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            artic_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            artic_api.CreateSleepThresholdAttr().Set(0.0)
            artic_api.CreateStabilizationThresholdAttr().Set(0.0)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb_api.CreateSleepThresholdAttr().Set(0.0)


async def setup_ant_scene(physics_rate: float = 60.0) -> AntConfig:
    """Load the ant USD scene and return configuration data.

    Args:
        physics_rate: Physics simulation rate in Hz.

    Returns:
        AntConfig with paths and sensor configuration for the ant robot.
    """
    assets_root_path = await get_assets_root_path_async()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        raise RuntimeError("Could not find Isaac Sim assets folder")

    ant_usd = assets_root_path + "/Isaac/Robots/IsaacSim/Ant/ant_colored.usd"
    result, stage = await stage_utils.open_stage_async(ant_usd)
    if not result or stage is None:
        raise RuntimeError(f"Failed to open ant stage: {ant_usd}")
    await omni.kit.app.get_app().next_update_async()

    # Keep asset metersPerUnit; ant_colored is authored for that scale.
    config = AntConfig()
    author_ant_standing_drives(config, stage)
    disable_articulation_sleep("/Ant", stage=stage)
    SimulationManager.setup_simulation(dt=1.0 / physics_rate)

    return config
