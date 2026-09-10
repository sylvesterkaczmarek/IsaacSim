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

"""Run two independent manipulation controller trees in one simulation."""

from __future__ import annotations

import argparse
import math


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("franka", "ur10"), default="franka")
parser.add_argument("--test", action="store_true")
parser.add_argument("--test-steps", type=_positive_int, default=None)
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

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
from isaacsim.robot_motion.cumotion import RmpFlowController, load_cumotion_supported_robot
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

_PHYSICS_DT = 1.0 / 60.0
_OFFSETS = ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0))


def _controller(scenario: ManipulationScenario) -> PickPlaceController:
    config = scenario.robot_config
    gripper = config.gripper
    if isinstance(gripper, JointGripperConfig):
        common = {
            "robot_joint_space": scenario.joint_space,
            "joint_names": gripper.joint_names,
            "open_positions": gripper.open_positions,
            "closed_positions": gripper.closed_positions,
            "position_tolerance": gripper.position_tolerance,
            "velocity_tolerance": gripper.velocity_tolerance,
            "minimum_close_fraction": gripper.minimum_close_fraction,
        }
        open_gripper: mg.BaseController = JointGripperController(**common, command=GripperCommand.OPEN)
        close_gripper: mg.BaseController = JointGripperController(**common, command=GripperCommand.CLOSE)
    elif isinstance(gripper, SurfaceGripperConfig):
        path = f"{scenario.robot_prim_path}/{gripper.relative_path}"
        open_gripper = SurfaceGripperController(gripper_path=path, command=GripperCommand.OPEN)
        close_gripper = SurfaceGripperController(gripper_path=path, command=GripperCommand.CLOSE)
    else:
        raise TypeError(f"Unsupported gripper configuration: {type(gripper).__name__}")

    robot = load_cumotion_supported_robot(config.name)
    tool_frame = config.tool.controller_frame
    supported_frames = robot.robot_description.tool_frame_names()
    if tool_frame not in supported_frames:
        raise RuntimeError(f"cuMotion configuration for {config.name!r} does not support tool frame {tool_frame!r}.")
    arm = RmpFlowController(
        cumotion_robot=robot,
        cumotion_world_interface=scenario.world_interface,
        robot_joint_space=scenario.joint_space,
        robot_site_space=scenario.site_space,
        tool_frame=tool_frame,
    )
    return PickPlaceController(
        arm_controller=arm,
        gripper_open_controller=open_gripper,
        gripper_close_controller=close_gripper,
        robot_site_space=scenario.site_space,
        tool_frame=tool_frame,
        controller_to_grasp_position=config.tool.controller_to_grasp_position,
        controller_to_grasp_orientation=config.tool.controller_to_grasp_orientation,
        grasp_orientation=config.grasp_orientation,
        approach_height=0.30 if isinstance(gripper, SurfaceGripperConfig) else 0.20,
    )


def _goal(pick: tuple[float, float, float], place: tuple[float, float, float]) -> mg.RobotState:
    names = [PickPlaceController.PICK_SITE, PickPlaceController.PLACE_SITE]
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=names,
            positions=(names, wp.array([pick, place], dtype=wp.float32, device="cpu")),
        )
    )


def _default_test_steps(controller: PickPlaceController) -> int:
    return math.ceil(sum(controller.phase_timeouts.values()) / _PHYSICS_DT) + math.ceil(0.5 / _PHYSICS_DT)


def _object_state(cube: RigidPrim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = cube.get_world_poses()[0].numpy()[0]
    linear_velocity, angular_velocity = cube.get_velocities()
    return position, linear_velocity.numpy()[0], angular_velocity.numpy()[0]


def _objects_are_settled(cubes: list[RigidPrim], places: list[tuple[float, float, float]]) -> bool:
    for cube, place in zip(cubes, places):
        position, linear_velocity, angular_velocity = _object_state(cube)
        target = np.asarray(place)
        if (
            np.linalg.norm(position[:2] - target[:2]) > 0.05
            or abs(position[2] - target[2]) > 0.03
            or np.linalg.norm(linear_velocity) > 0.05
            or np.linalg.norm(angular_velocity) > 0.5
        ):
            return False
    return True


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
    """Create two robots and advance both pick/place tasks each frame."""
    SimulationManager.setup_simulation(dt=_PHYSICS_DT, device="cpu")
    scenarios: list[ManipulationScenario] = []
    goals: list[mg.RobotState] = []
    cube_paths: list[str] = []
    cubes: list[RigidPrim] = []
    picks: list[tuple[float, float, float]] = []
    places: list[tuple[float, float, float]] = []

    for index, offset in enumerate(_OFFSETS):
        scenario = ManipulationScenario(args.robot, robot_prim_path=f"/World/robot_{index}", offset=offset)
        scenario.setup_scene()
        pick = (0.45, offset[1], 0.025)
        place = (0.45, offset[1] + 0.30, 0.025)
        cube_path = f"/World/Cube_{index}"
        cube = Cube(cube_path, positions=[pick], sizes=0.05)
        cubes.append(RigidPrim(cube.paths))
        GeomPrim(cube.paths, apply_collision_apis=True)
        grasp_height = 0.025 if isinstance(scenario.robot_config.gripper, SurfaceGripperConfig) else 0.0
        scenarios.append(scenario)
        goals.append(
            _goal(
                (pick[0], pick[1], pick[2] + grasp_height),
                (place[0], place[1], place[2] + grasp_height),
            )
        )
        cube_paths.append(cube_path)
        picks.append(pick)
        places.append(place)

    robot_paths = [scenario.robot_prim_path for scenario in scenarios]
    for index, scenario in enumerate(scenarios):
        other_cube_paths = [path for cube_index, path in enumerate(cube_paths) if cube_index != index]
        scenario.initialize_world_binding(exclude_prim_paths=(*robot_paths, *other_cube_paths))
    surface_gripper = isinstance(scenarios[0].robot_config.gripper, SurfaceGripperConfig)
    if surface_gripper:
        for scenario, cube_path in zip(scenarios, cube_paths):
            scenario.set_planning_obstacles_enabled((cube_path,), False)
    controllers = [_controller(scenario) for scenario in scenarios]
    step_cap = (args.test_steps or _default_test_steps(controllers[0])) if args.test else None

    simulation_app.update()
    app_utils.play()
    simulation_app.update()

    controller_time = 0.0
    for scenario, controller, goal in zip(scenarios, controllers, goals):
        if not controller.reset(scenario.read_robot_state(), goal, t=controller_time):
            raise RuntimeError(controller.failure_reason or "PickPlaceController reset failed.")

    steps = 0
    maximum_heights = [pick[2] for pick in picks]
    obstacles_disabled = [surface_gripper] * len(scenarios)
    try:
        while simulation_app.is_running() and not all(controller.is_done for controller in controllers):
            simulation_app.update()
            if app_utils.is_playing() and SimulationManager.is_simulating():
                controller_time += _PHYSICS_DT
                for index, (scenario, controller, goal) in enumerate(zip(scenarios, controllers, goals)):
                    if controller.failed:
                        raise RuntimeError(controller.failure_reason or f"Controller {index} failed.")
                    if controller.is_done:
                        continue
                    scenario.sync_world()
                    previous_phase = controller.phase
                    desired = controller.forward(scenario.read_robot_state(), goal, controller_time)
                    if desired is None:
                        raise RuntimeError(controller.failure_reason or f"Controller {index} returned no command.")
                    scenario.apply_robot_state(desired)
                    if previous_phase is not controller.phase:
                        if controller.phase is PickPlacePhase.LIFT:
                            _require_surface_attachment(scenario, cube_paths[index])
                        if controller.phase is PickPlacePhase.DESCEND_PICK:
                            scenario.set_planning_obstacles_enabled((cube_paths[index],), False)
                            obstacles_disabled[index] = True
                        elif controller.phase is PickPlacePhase.RETREAT:
                            scenario.set_planning_obstacles_enabled((cube_paths[index],), True)
                            obstacles_disabled[index] = False
                    maximum_heights[index] = max(
                        maximum_heights[index], float(cubes[index].get_world_poses()[0].numpy()[0, 2])
                    )
                steps += 1
                if (
                    step_cap is not None
                    and steps >= step_cap
                    and not all(controller.is_done for controller in controllers)
                ):
                    raise TimeoutError(f"Multiple tasks did not complete within {step_cap} simulation steps.")

        if not all(controller.is_done for controller in controllers):
            raise RuntimeError("Simulation stopped before both manipulation tasks completed.")
        for index, (maximum_height, pick) in enumerate(zip(maximum_heights, picks)):
            if maximum_height < pick[2] + 0.05:
                raise RuntimeError(f"Cube {index} did not satisfy the 0.05 m lift gate.")

        stable_steps = 0
        required_stable_steps = math.ceil(0.5 / _PHYSICS_DT)
        while simulation_app.is_running() and stable_steps < required_stable_steps:
            simulation_app.update()
            if app_utils.is_playing() and SimulationManager.is_simulating():
                stable_steps = stable_steps + 1 if _objects_are_settled(cubes, places) else 0
                steps += 1
                if step_cap is not None and steps >= step_cap and stable_steps < required_stable_steps:
                    raise TimeoutError(f"Multiple-task objects did not settle within {step_cap} simulation steps.")
        if stable_steps < required_stable_steps:
            raise RuntimeError("Simulation stopped before multiple-task placement validation completed.")
    finally:
        for index, (scenario, disabled) in enumerate(zip(scenarios, obstacles_disabled)):
            if disabled:
                scenario.set_planning_obstacles_enabled((cube_paths[index],), True)

    if args.test:
        print(f"PASS: {args.robot} completed two independent pick/place tasks.")
    else:
        print("Both manipulation tasks completed")


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
