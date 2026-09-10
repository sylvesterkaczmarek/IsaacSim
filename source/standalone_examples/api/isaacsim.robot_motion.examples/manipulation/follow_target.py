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

"""Track a draggable target with Franka or UR10 using cuMotion RMPflow."""

from __future__ import annotations

import argparse


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("franka", "ur10"), default="franka")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
parser.add_argument("--with-obstacle", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--test-steps", type=_positive_int, default=900)
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
from isaacsim.core.experimental.prims import GeomPrim
from isaacsim.core.experimental.utils import transform as transform_utils
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.cumotion import RmpFlowController, load_cumotion_supported_robot
from isaacsim.robot_motion.examples.manipulation import ManipulationScenario

_TARGET_PATH = "/World/TargetCube"
_OBSTACLE_PATH = "/World/Obstacle"
_PHYSICS_DT = 1.0 / 60.0


def _target_state(scenario: ManipulationScenario, target: GeomPrim) -> mg.RobotState:
    """Return the current target pose as a tool-site setpoint."""
    positions, _ = target.get_world_poses()
    grasp_position = positions.numpy()[0]
    grasp_orientation = scenario.robot_config.grasp_orientation
    tool = scenario.robot_config.tool
    grasp_to_controller_orientation = transform_utils.quaternion_conjugate(
        tool.controller_to_grasp_orientation, dtype=wp.float32, device=positions.device
    )
    grasp_to_controller_position = transform_utils.rotate_vectors_by_quaternion(
        -np.asarray(tool.controller_to_grasp_position),
        grasp_to_controller_orientation,
        dtype=wp.float32,
        device=positions.device,
    )
    controller_position = transform_utils.transform_local_to_world(
        grasp_to_controller_position,
        grasp_position,
        grasp_orientation,
        dtype=wp.float32,
        device=positions.device,
    )
    controller_orientation = transform_utils.quaternion_multiplication(
        grasp_orientation,
        grasp_to_controller_orientation,
        dtype=wp.float32,
        device=positions.device,
    )
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=scenario.site_space,
            positions=(
                [scenario.robot_config.tool.controller_frame],
                controller_position.reshape((1, 3)),
            ),
            orientations=(
                [scenario.robot_config.tool.controller_frame],
                controller_orientation.reshape((1, 4)),
            ),
        )
    )


def _target_errors(estimated: mg.RobotState, setpoint: mg.RobotState, tool_frame: str) -> tuple[float, float] | None:
    if estimated.sites is None or setpoint.sites is None:
        return None
    measured = estimated.sites
    target = setpoint.sites
    if (
        measured.positions is None
        or measured.orientations is None
        or target.positions is None
        or target.orientations is None
    ):
        return None
    measured_position = measured.positions.numpy()[measured.position_names.index(tool_frame)]
    target_position = target.positions.numpy()[target.position_names.index(tool_frame)]
    measured_orientation = measured.orientations.numpy()[measured.orientation_names.index(tool_frame)]
    target_orientation = target.orientations.numpy()[target.orientation_names.index(tool_frame)]
    dot = abs(float(np.dot(measured_orientation, target_orientation)))
    orientation_error = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
    return float(np.linalg.norm(measured_position - target_position)), float(orientation_error)


def _target_converged(estimated: mg.RobotState, setpoint: mg.RobotState, tool_frame: str) -> bool:
    errors = _target_errors(estimated, setpoint, tool_frame)
    return errors is not None and errors[0] <= 0.03 and errors[1] <= 0.15


def main() -> None:
    """Run the generic follow-target example."""
    SimulationManager.setup_simulation(dt=_PHYSICS_DT, device=args.device)

    scenario = ManipulationScenario(args.robot)
    scenario.setup_scene()
    target_cube = Cube(_TARGET_PATH, positions=[0.5, 0.0, 0.35], sizes=0.05, colors=[1.0, 0.0, 0.0])
    target = GeomPrim(target_cube.paths)

    excluded_paths = [_TARGET_PATH]
    if args.with_obstacle:
        Cube(_OBSTACLE_PATH, positions=[0.35, 0.0, 0.35], sizes=0.08, colors=[0.2, 0.2, 0.8])
        GeomPrim(_OBSTACLE_PATH, apply_collision_apis=True)

    scenario.initialize_world_binding(exclude_prim_paths=excluded_paths)
    cumotion_robot = load_cumotion_supported_robot(args.robot)
    tool_frame = scenario.robot_config.tool.controller_frame
    supported_frames = cumotion_robot.robot_description.tool_frame_names()
    if tool_frame not in supported_frames:
        raise RuntimeError(f"cuMotion configuration for {args.robot!r} does not support tool frame {tool_frame!r}.")
    controller = RmpFlowController(
        cumotion_robot=cumotion_robot,
        cumotion_world_interface=scenario.world_interface,
        robot_joint_space=scenario.joint_space,
        robot_site_space=scenario.site_space,
        tool_frame=tool_frame,
    )

    simulation_app.update()
    app_utils.play()
    simulation_app.update()

    print(f"{args.robot} follow-target (cuMotion RMPflow)")
    print(f"  Select {_TARGET_PATH} and drag it while playing.")

    reset_needed = True
    steps = 0
    stable_steps = 0
    passed = False
    controller_time = 0.0
    while simulation_app.is_running():
        simulation_app.update()
        if app_utils.is_playing() and SimulationManager.is_simulating():
            estimated = scenario.read_robot_state()
            setpoint = _target_state(scenario, target)
            if reset_needed:
                controller_time = 0.0
                if not controller.reset(estimated, setpoint, t=controller_time):
                    raise RuntimeError("RmpFlowController reset failed.")
                reset_needed = False
            else:
                controller_time += _PHYSICS_DT
                scenario.sync_world()
                desired = controller.forward(estimated, setpoint, controller_time)
                if desired is None:
                    raise RuntimeError("RmpFlowController returned no command.")
                scenario.apply_robot_state(desired)
            if args.test:
                stable_steps = stable_steps + 1 if _target_converged(estimated, setpoint, tool_frame) else 0
                steps += 1
                if stable_steps >= 10:
                    passed = True
                    break
                if steps >= args.test_steps:
                    errors = _target_errors(estimated, setpoint, tool_frame)
                    detail = (
                        ""
                        if errors is None
                        else f" Position error: {errors[0]:.4f} m; orientation error: {errors[1]:.4f} rad."
                    )
                    raise TimeoutError(f"Target did not converge within {args.test_steps} simulation steps.{detail}")
        else:
            reset_needed = True
    if args.test:
        if not passed:
            raise RuntimeError("Simulation stopped before follow-target validation completed.")
        print(f"PASS: {args.robot} follow-target converged.")


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
