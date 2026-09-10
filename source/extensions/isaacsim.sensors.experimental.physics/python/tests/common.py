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

"""Provides shared fixtures, tolerances, and scene helpers for experimental physics sensor tests. Covers timeline reset helpers, gravity constants, and reusable ant and cube scene configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.app
import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

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
        seconds: Duration to step the simulation, in seconds.
    """
    dt = SimulationManager.get_physics_dt()
    steps = max(1, int(round(seconds / dt)))
    for _ in range(steps):
        await omni.kit.app.get_app().next_update_async()


async def reset_timeline(timeline: Any = None, *, steps: int = 2) -> None:
    """Stop and restart the timeline.

    Args:
        timeline: Timeline interface to reset. Uses the current timeline when ``None``.
        steps: Number of app updates to wait after restarting.
    """
    if timeline is None:
        timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()
    timeline.play()
    for _ in range(steps):
        await omni.kit.app.get_app().next_update_async()


# USD path the ant asset is referenced under when the test scene is assembled.
# Shared by the AntConfig path defaults and setup_ant_scene.
ANT_ROBOT_PATH = "/Ant"


@dataclass
class AntConfig:
    """Configuration data for ant robot used in sensor tests."""

    leg_paths: list[str] = field(
        default_factory=lambda: [
            f"{ANT_ROBOT_PATH}/{foot}"
            for foot in ("front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot")
        ]
    )
    sphere_path: str = f"{ANT_ROBOT_PATH}/torso"
    # Contact-sensor translation per foot, placed at the ankle capsule tip (the
    # ground-contact point) in the foot body's local frame so the sensor sphere
    # reaches the contact when the ant stands.
    sensor_offsets: list[np.ndarray] = field(
        default_factory=lambda: [
            np.array([[0.44, 0.44, 0.0]]),
            np.array([[-0.44, 0.44, 0.0]]),
            np.array([[-0.44, -0.44, 0.0]]),
            np.array([[0.44, -0.44, 0.0]]),
        ]
    )
    # IMU sensor offsets (at origin for each sensor location)
    imu_sensor_offsets: list[np.ndarray] = field(
        default_factory=lambda: [np.array([[0.0, 0.0, 0.0]]) for _ in range(5)]
    )
    # IMU sensor orientations (identity quaternions, wxyz)
    sensor_quatd: list[np.ndarray] = field(default_factory=lambda: [np.array([[1.0, 0.0, 0.0, 0.0]]) for _ in range(5)])
    colors: list[tuple[float, float, float, float]] = field(
        default_factory=lambda: [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
    )


def is_physx_engine() -> bool:
    """Return whether the active physics engine is PhysX."""
    return SimulationManager.get_active_physics_engine() == "physx"


def is_cuda_sim_available() -> bool:
    """Return whether a CUDA device is available for physics simulation."""
    try:
        import warp as wp

        return wp.is_cuda_available()
    except ImportError:
        return False


def author_ant_standing_drives(robot_path: str, drive_stiffness: float = 10.0, drive_damping: float = 1.0) -> None:
    """Author angular position drives that hold the ant in a standing pose.

    The ``ant.usd`` asset ships as a torque-controlled articulation with no joint
    drives, so left un-driven it collapses under gravity. This adds an angular
    drive with a standing-pose target on every revolute joint; the drives pull
    the ant from its rest pose into a stable stance under any engine.

    Args:
        robot_path: USD path to the ant robot root prim.
        drive_stiffness: Angular drive stiffness.
        drive_damping: Angular drive damping.
    """
    stage = stage_utils.get_current_stage()
    if stage is None:
        return
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        return

    # Standing pose the drives hold, in degrees, keyed by joint name. The leg
    # joints stay at zero while the foot joints splay outward so the four ankle
    # capsules rest on the ground. Values match the multiphysics ant so both
    # PhysX and Newton settle into the same pose.
    standing_pose_deg = {
        "front_left_leg": 0.0,
        "front_left_foot": 55.0,
        "front_right_leg": 0.0,
        "front_right_foot": -55.0,
        "left_back_leg": 0.0,
        "left_back_foot": -55.0,
        "right_back_leg": 0.0,
        "right_back_foot": 55.0,
    }
    # Keep the articulation awake so resting foot contacts continue to be
    # reported; a settled articulation otherwise sleeps and stops emitting them.
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    drives_applied = 0
    for prim in Usd.PrimRange(robot_prim):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        target_deg = standing_pose_deg.get(prim.GetName())
        if target_deg is None:
            carb.log_warn(f"author_ant_standing_drives: unrecognized joint '{prim.GetName()}' at {prim.GetPath()}")
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
        drive.CreateStiffnessAttr().Set(drive_stiffness)
        drive.CreateDampingAttr().Set(drive_damping)
        drive.CreateTargetPositionAttr().Set(target_deg)
        drives_applied += 1
    if drives_applied == 0:
        carb.log_warn(f"author_ant_standing_drives: no drives applied under '{robot_path}'")


async def setup_ant_scene(
    physics_rate: float = 60.0, drive_stiffness: float = 10.0, drive_damping: float = 1.0
) -> AntConfig:
    """Assemble the ant test scene and return its configuration data.

    Builds a fresh stage with a ground plane, references the ant robot, and
    authors the standing-pose drives so the ant rests on its feet under any
    engine.

    Args:
        physics_rate: Physics simulation rate in Hz.
        drive_stiffness: Angular drive stiffness for standing pose.
        drive_damping: Angular drive damping for standing pose.

    Returns:
        AntConfig with paths and sensor configuration for the ant robot.
    """
    from isaacsim.core.experimental.objects import GroundPlane
    from isaacsim.core.experimental.utils.stage import add_reference_to_stage

    assets_root_path = await get_assets_root_path_async()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        raise RuntimeError("Could not find Isaac Sim assets folder")

    await stage_utils.create_new_stage_async()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    GroundPlane("/World/GroundPlane", sizes=10.0)
    add_reference_to_stage(usd_path=assets_root_path + "/Isaac/Robots/IsaacSim/Ant/ant.usd", path=ANT_ROBOT_PATH)
    await omni.kit.app.get_app().next_update_async()

    # Lift the ant to its standing height before simulation: the asset authors the
    # torso at z=0, burying the feet, and Newton resolves that penetration by
    # launching the ant airborne.
    stage = stage_utils.get_current_stage()
    ant_prim = stage.GetPrimAtPath(ANT_ROBOT_PATH)
    UsdGeom.Xformable(ant_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.6))
    stage.SetDefaultPrim(ant_prim)

    author_ant_standing_drives(ANT_ROBOT_PATH, drive_stiffness=drive_stiffness, drive_damping=drive_damping)
    SimulationManager.setup_simulation(dt=1.0 / physics_rate)

    return AntConfig()
