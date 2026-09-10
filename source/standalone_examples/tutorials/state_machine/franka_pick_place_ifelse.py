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

"""Franka reactive stacking with no state machine: one controller does everything."""

from __future__ import annotations

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless, "hide_ui": False})

if args.headless:
    from isaacsim.core.experimental.utils.app import enable_extension

    simulation_app.set_setting("/app/window/drawMouse", True)
    enable_extension("omni.kit.livestream.app")

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from franka_stacking import (
    ControllerResources,
    FrankaStackingSceneIO,
    Observation,
    TaskConfig,
    make_task_config,
)
from isaacsim.core.simulation_manager import SimulationManager

_PHYSICS_DT = 1.0 / 60.0

# Phases run in this fixed order, addressed by an integer counter.
_PHASES = ("pre_grasp", "approach", "grasp", "lift", "transport", "lower", "release", "retract")
# Phases where the gripper is closed around the cube.
_CLOSED_PHASES = {"grasp", "lift", "transport", "lower"}
# Phases counted as "holding a cube" for the drop and stack-health checks.
_HOLDING_PHASES = {3, 4, 5}
# Phases that finish quickly, so they get a shorter timeout.
_SHORT_PHASES = {1, 3, 4, 5}

# Franka-specific facts the controller needs to command the hand and aim the tool.
_FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
_DOWN_ORIENTATION = np.array([0.0, 0.0, 1.0, 0.0])


class FrankaStackingIfElseController(mg.BaseController):
    """Stack four cubes with one monolithic controller and no state machine.

    A single ``forward()`` runs the whole task: it decides what to do with an
    ``if/elif`` chain over an integer phase counter, computes the arm target and
    the gripper command inline, drives the RMPflow arm controller directly, and
    merges the arm and finger joint targets into one command. Task context lives
    in scattered instance attributes; every reactive check and every phase is
    another branch wedged into the same method.
    """

    def __init__(self, config: TaskConfig, resources: ControllerResources) -> None:
        self._config = config
        self._arm_controller = resources.arm_controller
        self._joint_space = resources.joint_space
        self._site_space = resources.site_space
        self._tool_frame = resources.tool_frame
        self._finger_names = [name for name in _FINGER_JOINTS if name in self._joint_space]
        # Scattered task context, no single owner:
        self._color: str | None = None
        self._destination: np.ndarray | None = None
        self._phase = 0
        self._phase_started = 0.0
        self._settle_started: float | None = None
        self._pickup_position: np.ndarray | None = None
        self._cube_reference_xy: np.ndarray | None = None
        self._aborted = False
        # Cached command state, used to grade progress on the next tick:
        self._target: np.ndarray | None = None
        self._gripper_open: bool | None = None
        self.done = False

    def reset(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> bool:
        """Clear task context and reset the arm controller."""
        self._color = None
        self._target = None
        self._gripper_open = None
        self.done = False
        return self._arm_controller.reset(estimated_state, setpoint_state, t, **kwargs)

    # DOCS: BEGIN forward_ifelse
    def forward(
        self,
        estimated_state: mg.RobotState,
        setpoint_state: mg.RobotState | None,
        t: float,
        *,
        observation: Observation,
        **kwargs: object,
    ) -> mg.RobotState | None:
        """Decide, react, and drive the arm and hand -- all in one method."""
        target_reached = self._target_reached(observation)
        gripper_reached = self._gripper_reached(observation)

        # Decide what to work on, react to the world, and advance the phase.
        if (self._color is None or self._phase >= len(_PHASES)) and not self._select_next_cube(observation):
            self.done = True
            self._color = None
        elif self._cube_was_dropped(observation):
            print(f"[drop] {self._color} fell out of gripper, re-decide")
            self._color = None
        else:
            self._restart_pickup_if_cube_moved(observation)
            self._advance_if_ready(observation, target_reached, gripper_reached)

        # Turn the current phase into an arm target and a gripper command, run
        # the arm controller, and merge the finger command into its output.
        phase = _PHASES[self._phase] if self._is_active() else "release"
        self._target = self._compute_target(observation) if self._is_active() else None
        self._gripper_open = phase not in _CLOSED_PHASES
        setpoint = None if self._target is None else self._arm_setpoint(self._target)
        arm_state = self._arm_controller.forward(estimated_state, setpoint, t, **kwargs)
        gripper_state = self._gripper_command(self._gripper_open)
        out_state = mg.RobotState()
        for state in (arm_state, gripper_state):
            out_state = mg.combine_robot_states(out_state, state)
            if out_state is None:
                return None
        return out_state

    # DOCS: END forward_ifelse

    def _is_active(self) -> bool:
        return self._color is not None and self._phase < len(_PHASES)

    def _select_next_cube(self, observation: Observation) -> bool:
        for color in self._config.cube_order:
            if not self._at_goal(observation.cube_positions[color], self._config.stack_positions[color]):
                self._color = color
                self._destination = self._config.stack_positions[color].copy()
                self._phase = 0
                self._phase_started = observation.time
                self._settle_started = None
                self._pickup_position = None
                self._cube_reference_xy = None
                self._aborted = False
                print(f"[decide] -> stack {color}")
                return True
        self._color = None
        return False

    def _cube_was_dropped(self, observation: Observation) -> bool:
        assert self._color is not None
        return self._phase in _HOLDING_PHASES and bool(
            np.linalg.norm(observation.cube_positions[self._color] - observation.end_effector_position)
            > self._config.drop_tolerance
        )

    def _restart_pickup_if_cube_moved(self, observation: Observation) -> None:
        assert self._color is not None
        if self._phase in _HOLDING_PHASES or self._pickup_position is not None:
            return
        current_xy = observation.cube_positions[self._color][:2]
        if self._cube_reference_xy is None:
            self._cube_reference_xy = current_xy.copy()
        elif np.linalg.norm(current_xy - self._cube_reference_xy) > self._config.cube_move_tolerance:
            print(f"[resync] {self._color} moved, back to pre-grasp")
            self._phase = 0
            self._phase_started = observation.time
            self._settle_started = None
            self._cube_reference_xy = current_xy.copy()

    def _advance_if_ready(self, observation: Observation, target_reached: bool, gripper_reached: bool) -> None:
        assert self._color is not None
        elapsed = max(0.0, observation.time - self._phase_started)
        advance = False
        reason = "ok"
        if self._phase == 2:
            advance = elapsed >= self._config.minimum_duration
        elif self._phase == 6:
            advance = elapsed >= self._config.minimum_duration and gripper_reached
        else:
            if target_reached:
                if self._settle_started is None:
                    self._settle_started = observation.time
            else:
                self._settle_started = None
            settled = (
                self._settle_started is not None
                and observation.time - self._settle_started >= self._config.settle_duration
            )
            timeout = self._config.short_phase_timeout if self._phase in _SHORT_PHASES else self._config.phase_timeout
            if elapsed >= self._config.minimum_duration and settled:
                advance, reason = True, "settled"
            elif elapsed >= timeout:
                advance, reason = True, "timeout"

        if not advance:
            return
        print(f"  [{self._color}] phase {self._phase} done ({reason})")
        self._phase += 1
        self._phase_started = observation.time
        self._settle_started = None
        if self._phase == 2:
            self._pickup_position = observation.cube_positions[self._color].copy()
        self._return_to_spawn_if_stack_broken(observation)

    def _return_to_spawn_if_stack_broken(self, observation: Observation) -> None:
        assert self._color is not None
        if self._phase not in _HOLDING_PHASES or self._aborted:
            return
        index = self._config.cube_order.index(self._color)
        for below in self._config.cube_order[:index]:
            if not self._at_goal(observation.cube_positions[below], self._config.stack_positions[below]):
                print(f"[abort] stack invalid below {self._color}, returning to spawn")
                self._destination = self._config.spawn_positions[self._color].copy()
                self._aborted = True
                return

    def _compute_target(self, observation: Observation) -> np.ndarray:
        assert self._color is not None and self._destination is not None
        source = self._pickup_position
        if source is None:
            source = observation.cube_positions[self._color]
        high = self._config.above_height
        low = self._config.cube_size / 2 + self._config.grasp_offset
        phase = _PHASES[self._phase]
        if phase in {"pre_grasp", "lift"}:
            target = np.array([source[0], source[1], source[2] + high])
        elif phase in {"approach", "grasp"}:
            target = np.array([source[0], source[1], source[2] + low])
        elif phase in {"transport", "retract"}:
            target = np.array([self._destination[0], self._destination[1], self._destination[2] + high])
        else:
            target = np.array([self._destination[0], self._destination[1], self._destination[2] + low])
        target[1] += self._config.tool_y_offset
        return target

    def _arm_setpoint(self, position: np.ndarray) -> mg.RobotState:
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=self._site_space,
                positions=([self._tool_frame], wp.array([position.tolist()], dtype=wp.float32)),
                orientations=([self._tool_frame], wp.array([_DOWN_ORIENTATION.tolist()], dtype=wp.float32)),
            )
        )

    def _gripper_command(self, gripper_open: bool) -> mg.RobotState:
        position = self._config.open_position if gripper_open else self._config.closed_position
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=self._joint_space,
                positions=(self._finger_names, wp.array([position] * len(self._finger_names), dtype=wp.float32)),
            )
        )

    def _target_reached(self, observation: Observation) -> bool:
        return self._target is not None and bool(
            np.linalg.norm(observation.end_effector_position - self._target) < self._config.end_effector_tolerance
        )

    def _gripper_reached(self, observation: Observation) -> bool:
        if self._gripper_open is None:
            return False
        gripper_target = self._config.open_position if self._gripper_open else self._config.closed_position
        return abs(observation.gripper_position - gripper_target) < self._config.gripper_tolerance

    def _at_goal(self, position: np.ndarray, goal: np.ndarray) -> bool:
        return bool(np.linalg.norm(position - goal) < self._config.goal_tolerance)


def main(args: argparse.Namespace, app: SimulationApp) -> None:
    """Run the state-machine-free stacking example."""
    SimulationManager.setup_simulation(dt=_PHYSICS_DT, device="cpu")
    config = make_task_config()
    scene = FrankaStackingSceneIO(config)
    controller: FrankaStackingIfElseController | None = None
    app.run_coroutine(scene.setup())
    app.update()

    if args.headless and not args.test:
        print("Headless: press Play in the livestream UI to begin.")
        initialized = False
    else:
        app_utils.play()
        app.update()
        scene.initialize()
        controller = FrankaStackingIfElseController(config, scene.controller_resources())
        initialized = True

    needs_reset = True
    frame_count = 0
    while app.is_running():
        app.update()
        if app_utils.is_playing() and SimulationManager.is_simulating():
            if not initialized:
                scene.initialize()
                controller = FrankaStackingIfElseController(config, scene.controller_resources())
                initialized = True
            assert controller is not None
            if needs_reset:
                scene.reset()
                time = SimulationManager.get_simulation_time()
                estimated, observation = scene.read(time)
                if not controller.reset(estimated, None, time):
                    raise RuntimeError("Controller reset failed")
                desired = controller.forward(estimated, None, time, observation=observation)
                scene.apply(desired)
                needs_reset = False
                frame_count = 0
                continue

            time = SimulationManager.get_simulation_time()
            estimated, observation = scene.read(time)
            desired = controller.forward(estimated, None, time, observation=observation)
            scene.apply(desired)
            frame_count += 1
            if controller.done:
                print("Stacking complete.")
                break
            if args.test and frame_count >= 2400:
                raise RuntimeError("Timed out before stacking completed.")
        elif app_utils.is_stopped():
            needs_reset = True


if __name__ == "__main__":
    try:
        main(args, simulation_app)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        simulation_app.close()
