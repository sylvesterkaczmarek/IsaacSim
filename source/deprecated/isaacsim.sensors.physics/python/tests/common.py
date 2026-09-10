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

"""Common test utilities and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, PhysxSchema, Usd, UsdPhysics

EARTH_GRAVITY = 9.81
MOON_GRAVITY = 1.62
CM_GRAVITY = 981.0

GRAVITY_TOLERANCE = 0.1
ANGLE_TOLERANCE_DEG = 0.1
ANGULAR_VEL_TOLERANCE = 0.2
ORIENTATION_TOLERANCE = 1e-4
SMALL_TOLERANCE = 0.01


async def step_simulation(seconds: float) -> None:
    """Step the simulation forward by the given number of seconds.

    Args:
        seconds: Number of seconds to simulate.
    """
    dt = SimulationManager.get_physics_dt()
    steps = max(1, int(round(seconds / dt)))
    for _ in range(steps):
        await omni.kit.app.get_app().next_update_async()


async def reset_timeline(timeline: Any = None, *, steps: int = 2) -> None:
    """Stop and restart the timeline.

    Args:
        timeline: Timeline interface to control. If None, uses the global timeline.
        steps: Number of simulation steps to run after restarting.
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
    """Configuration data for the ant robot used in sensor tests.

    Args:
        leg_paths: Prim paths for the ant lower arm links.
        sphere_path: Prim path for the ant sphere body.
        sensor_offsets: Sensor offsets for the leg sensors.
        imu_sensor_offsets: Sensor offsets for the IMU sensors.
        sensor_quatd: Sensor orientation quaternions for the IMU sensors.
        colors: RGBA colors for the leg sensors.
        shoulder_joints: Prim paths for the ant shoulder joints.
        lower_joints: Prim paths for the ant lower arm joints.
    """

    leg_paths: list[str] = field(default_factory=lambda: [f"/Ant/Arm_{i + 1:02d}/Lower_Arm" for i in range(4)])
    sphere_path: str = "/Ant/Sphere"
    """Path to the ant sphere prim."""
    sensor_offsets: list[Gf.Vec3d] = field(default_factory=lambda: [Gf.Vec3d(40, 0, 0) for _ in range(4)])
    # IMU sensor offsets (at origin for each sensor location)
    imu_sensor_offsets: list[Gf.Vec3d] = field(default_factory=lambda: [Gf.Vec3d(0, 0, 0) for _ in range(5)])
    # IMU sensor orientations (identity quaternions)
    sensor_quatd: list[Gf.Quatd] = field(default_factory=lambda: [Gf.Quatd(1, 0, 0, 0) for _ in range(5)])
    colors: list[tuple[float, float, float, float]] = field(
        default_factory=lambda: [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
    )
    shoulder_joints: list[str] = field(
        default_factory=lambda: [f"/Ant/Arm_{i + 1:02d}/Upper_Arm/shoulder_joint" for i in range(4)]
    )
    lower_joints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass creation."""
        if not self.lower_joints:
            self.lower_joints = [f"{path}/lower_arm_joint" for path in self.leg_paths]


def disable_articulation_sleep(robot_path: str = "/Ant", stage: Usd.Stage | None = None) -> None:
    """Disable sleep and stabilization thresholds under the robot.

    Args:
        robot_path: USD path to the robot root prim.
        stage: Stage to modify. Uses the current USD context stage when ``None``.
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
            articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            articulation_api.CreateSleepThresholdAttr().Set(0.0)
            articulation_api.CreateStabilizationThresholdAttr().Set(0.0)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid_body_api.CreateSleepThresholdAttr().Set(0.0)


async def setup_ant_scene(physics_rate: float = 60.0) -> AntConfig:
    """Load the ant USD scene and return configuration data.

    Args:
        physics_rate: Physics simulation rate in Hz.

    Returns:
        Paths and sensor configuration for the ant robot.

    Raises:
        RuntimeError: If the Isaac Sim assets folder cannot be found.
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

    disable_articulation_sleep("/Ant", stage=stage)
    SimulationManager.setup_simulation(dt=1.0 / physics_rate)

    return AntConfig()
