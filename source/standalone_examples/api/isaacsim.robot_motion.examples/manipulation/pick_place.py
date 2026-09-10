# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pick and place with Franka or UR10 using composable motion-generation controllers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("franka", "ur10"), default="franka")
parser.add_argument(
    "--usd-path",
    default=None,
    help="Optional USD override compatible with the selected robot configuration",
)
parser.add_argument(
    "--robot-config-dir",
    type=Path,
    default=None,
    help="Optional directory containing a custom URDF, XRDF, and rmp_flow.yaml",
)
parser.add_argument("--urdf", type=Path, default=None, help="URDF filename within --robot-config-dir (robot.urdf)")
parser.add_argument("--xrdf", type=Path, default=None, help="XRDF filename within --robot-config-dir (robot.xrdf)")
parser.add_argument("--test", action="store_true")
parser.add_argument("--test-steps", type=_positive_int, default=None)
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()
if args.robot_config_dir is None:
    if args.urdf is not None or args.xrdf is not None:
        parser.error("--urdf and --xrdf require --robot-config-dir")
else:
    args.robot_config_dir = args.robot_config_dir.expanduser().resolve()
    args.urdf = args.urdf or Path("robot.urdf")
    args.xrdf = args.xrdf or Path("robot.xrdf")
    if args.urdf.parent != Path() or args.xrdf.parent != Path():
        parser.error("--urdf and --xrdf must be filenames within --robot-config-dir")
    for filename in (args.urdf, args.xrdf, Path("rmp_flow.yaml")):
        if not (args.robot_config_dir / filename).is_file():
            parser.error(f"Required robot configuration file not found: {args.robot_config_dir / filename}")

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import omni.kit.app

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.robot_motion.examples", True)

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.cumotion import RmpFlowController, load_cumotion_robot, load_cumotion_supported_robot
from isaacsim.robot_motion.examples.manipulation import (
    GripperCommand,
    JointGripperConfig,
    JointGripperController,
    ManipulationScenario,
    PickPlaceController,
    PickPlacePhase,
    SurfaceGripperConfig,
    SurfaceGripperController,
)

_ROBOT_PATH = "/World/robot"
_CUBE_PATH = "/World/Cube"
_TARGET_PATH = "/World/PlaceTarget"
_PHYSICS_DT = 1.0 / 60.0


def _make_gripper_controllers(scenario: ManipulationScenario) -> tuple[mg.BaseController, mg.BaseController]:
    """Construct open and close controller leaves from the selected robot configuration."""
    gripper = scenario.robot_config.gripper
    if isinstance(gripper, JointGripperConfig):
        arguments = {
            "robot_joint_space": scenario.joint_space,
            "joint_names": gripper.joint_names,
            "open_positions": gripper.open_positions,
            "closed_positions": gripper.closed_positions,
            "position_tolerance": gripper.position_tolerance,
            "velocity_tolerance": gripper.velocity_tolerance,
            "minimum_close_fraction": gripper.minimum_close_fraction,
        }
        return (
            JointGripperController(**arguments, command=GripperCommand.OPEN),
            JointGripperController(**arguments, command=GripperCommand.CLOSE),
        )
    if isinstance(gripper, SurfaceGripperConfig):
        gripper_path = f"{scenario.robot_prim_path}/{gripper.relative_path}"
        return (
            SurfaceGripperController(gripper_path=gripper_path, command=GripperCommand.OPEN),
            SurfaceGripperController(gripper_path=gripper_path, command=GripperCommand.CLOSE),
        )
    raise TypeError(f"Unsupported gripper configuration: {type(gripper).__name__}")


def _make_goal(pick: tuple[float, float, float], place: tuple[float, float, float]) -> mg.RobotState:
    """Build the robot-independent named pick/place goal."""
    names = [PickPlaceController.PICK_SITE, PickPlaceController.PLACE_SITE]
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=names,
            positions=(names, wp.array([pick, place], dtype=wp.float32, device="cpu")),
        )
    )


def _make_controller(scenario: ManipulationScenario) -> PickPlaceController:
    """Construct RMPflow and compose it with robot-specific gripper leaves."""
    if args.robot_config_dir is None:
        cumotion_robot = load_cumotion_supported_robot(scenario.robot_config.name)
    else:
        cumotion_robot = load_cumotion_robot(
            directory=args.robot_config_dir,
            urdf_filename=args.urdf,
            xrdf_filename=args.xrdf,
        )
    tool_frame = scenario.robot_config.tool.controller_frame
    supported_frames = cumotion_robot.robot_description.tool_frame_names()
    if tool_frame not in supported_frames:
        raise RuntimeError(
            f"cuMotion configuration for {scenario.robot_config.name!r} does not support tool frame {tool_frame!r}."
        )
    arm = RmpFlowController(
        cumotion_robot=cumotion_robot,
        cumotion_world_interface=scenario.world_interface,
        robot_joint_space=scenario.joint_space,
        robot_site_space=scenario.site_space,
        tool_frame=tool_frame,
    )
    open_gripper, close_gripper = _make_gripper_controllers(scenario)
    return PickPlaceController(
        arm_controller=arm,
        gripper_open_controller=open_gripper,
        gripper_close_controller=close_gripper,
        robot_site_space=scenario.site_space,
        tool_frame=tool_frame,
        controller_to_grasp_position=scenario.robot_config.tool.controller_to_grasp_position,
        controller_to_grasp_orientation=scenario.robot_config.tool.controller_to_grasp_orientation,
        grasp_orientation=scenario.robot_config.grasp_orientation,
        approach_height=0.30 if isinstance(scenario.robot_config.gripper, SurfaceGripperConfig) else 0.20,
    )


def _test_step_cap(controller: PickPlaceController) -> int | None:
    if not args.test:
        return None
    if args.test_steps is not None:
        return args.test_steps
    return math.ceil(sum(controller.phase_timeouts.values()) / _PHYSICS_DT) + math.ceil(0.5 / _PHYSICS_DT)


def _object_state(cube: RigidPrim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = cube.get_world_poses()[0].numpy()[0]
    linear_velocity, angular_velocity = cube.get_velocities()
    return position, linear_velocity.numpy()[0], angular_velocity.numpy()[0]


def _is_settled(cube: RigidPrim, place: tuple[float, float, float]) -> bool:
    position, linear_velocity, angular_velocity = _object_state(cube)
    target = np.asarray(place)
    return (
        np.linalg.norm(position[:2] - target[:2]) <= 0.05
        and abs(position[2] - target[2]) <= 0.03
        and np.linalg.norm(linear_velocity) <= 0.05
        and np.linalg.norm(angular_velocity) <= 0.5
    )


def _require_surface_attachment(scenario: ManipulationScenario, object_path: str) -> None:
    gripper = scenario.robot_config.gripper
    if not isinstance(gripper, SurfaceGripperConfig):
        return
    from isaacsim.robot.surface_gripper import _surface_gripper

    gripper_path = f"{scenario.robot_prim_path}/{gripper.relative_path}"
    gripped_objects = _surface_gripper.acquire_surface_gripper_interface().get_gripped_objects(gripper_path)
    if not any(path == object_path or path.startswith(object_path + "/") for path in gripped_objects):
        raise RuntimeError(f"Surface gripper did not attach the active object {object_path}.")


def main() -> None:
    """Run one pick-and-place task."""
    SimulationManager.setup_simulation(dt=_PHYSICS_DT, device="cpu")
    scenario = ManipulationScenario(args.robot, robot_prim_path=_ROBOT_PATH, robot_usd_path=args.usd_path)
    scenario.setup_scene()

    pick = (0.45, 0.0, 0.025)
    place = (0.45, 0.35, 0.025)
    cube = Cube(_CUBE_PATH, positions=[pick], sizes=0.05, colors=[[0.8, 0.1, 0.1]])
    cube_prim = RigidPrim(cube.paths)
    GeomPrim(cube.paths, apply_collision_apis=True)
    Cube(_TARGET_PATH, positions=[place], sizes=0.055, colors=[[0.1, 0.8, 0.1]])

    scenario.initialize_world_binding(exclude_prim_paths=(_TARGET_PATH,))
    surface_gripper = isinstance(scenario.robot_config.gripper, SurfaceGripperConfig)
    if surface_gripper:
        scenario.set_planning_obstacles_enabled((_CUBE_PATH,), False)
    controller = _make_controller(scenario)
    grasp_height = 0.025 if surface_gripper else 0.0
    goal = _make_goal(
        (pick[0], pick[1], pick[2] + grasp_height),
        (place[0], place[1], place[2] + grasp_height),
    )
    step_cap = _test_step_cap(controller)

    simulation_app.update()
    app_utils.play()
    simulation_app.update()

    controller_time = 0.0
    estimated = scenario.read_robot_state()
    if not controller.reset(estimated, goal, t=controller_time):
        raise RuntimeError(controller.failure_reason or "PickPlaceController reset failed.")

    steps = 0
    maximum_height = pick[2]
    obstacle_disabled = surface_gripper
    try:
        while simulation_app.is_running() and not controller.is_done and not controller.failed:
            simulation_app.update()
            if app_utils.is_playing() and SimulationManager.is_simulating():
                controller_time += _PHYSICS_DT
                scenario.sync_world()
                previous_phase = controller.phase
                desired = controller.forward(scenario.read_robot_state(), goal, controller_time)
                if desired is None:
                    raise RuntimeError(controller.failure_reason or "PickPlaceController returned no command.")
                scenario.apply_robot_state(desired)
                if previous_phase is not controller.phase:
                    if controller.phase is PickPlacePhase.LIFT:
                        _require_surface_attachment(scenario, _CUBE_PATH)
                    if controller.phase is PickPlacePhase.DESCEND_PICK:
                        scenario.set_planning_obstacles_enabled((_CUBE_PATH,), False)
                        obstacle_disabled = True
                    elif controller.phase is PickPlacePhase.RETREAT:
                        scenario.set_planning_obstacles_enabled((_CUBE_PATH,), True)
                        obstacle_disabled = False
                maximum_height = max(maximum_height, float(cube_prim.get_world_poses()[0].numpy()[0, 2]))
                steps += 1
                if step_cap is not None and steps >= step_cap and not controller.is_done:
                    raise TimeoutError(f"Pick/place did not complete within {step_cap} simulation steps.")

        if controller.failed:
            raise RuntimeError(controller.failure_reason or "Pick-and-place failed.")
        if not controller.is_done:
            raise RuntimeError("Simulation stopped before pick/place completed.")
        if maximum_height < pick[2] + 0.05:
            raise RuntimeError("Cube did not satisfy the 0.05 m lift gate.")

        stable_steps = 0
        required_stable_steps = math.ceil(0.5 / _PHYSICS_DT)
        while simulation_app.is_running() and stable_steps < required_stable_steps:
            simulation_app.update()
            if app_utils.is_playing() and SimulationManager.is_simulating():
                stable_steps = stable_steps + 1 if _is_settled(cube_prim, place) else 0
                steps += 1
                if step_cap is not None and steps >= step_cap and stable_steps < required_stable_steps:
                    raise TimeoutError(f"Cube did not settle within {step_cap} simulation steps.")
        if stable_steps < required_stable_steps:
            raise RuntimeError("Simulation stopped before placement validation completed.")
    finally:
        if obstacle_disabled:
            scenario.set_planning_obstacles_enabled((_CUBE_PATH,), True)

    if args.test:
        print(f"PASS: {args.robot} pick/place completed with a settled placement.")
    else:
        print("Done picking and placing")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
    if exit_code:
        raise SystemExit(exit_code)
