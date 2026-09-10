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

"""Franka reactive stacking with a library-free finite state machine."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

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
import numpy as np
from franka_stacking import (
    FrankaStackingSceneIO,
    Observation,
    PickPlaceController,
    PickPlaceDecision,
    PickPlacePhase,
    TaskConfig,
    make_pick_place_controller,
    make_task_config,
)
from isaacsim.core.simulation_manager import SimulationManager

_PHYSICS_DT = 1.0 / 60.0


# DOCS: BEGIN task_class
@dataclass
class PickPlaceState:
    """State for one active cube transfer."""

    color: str
    destination: np.ndarray
    phase: PickPlacePhase
    phase_started: float
    settle_started: float | None = None
    pickup_position: np.ndarray | None = None
    cube_reference_xy: np.ndarray | None = None
    aborted: bool = False

    @property
    def holding(self) -> bool:
        return self.phase in {PickPlacePhase.LIFT, PickPlacePhase.TRANSPORT, PickPlacePhase.LOWER}


# DOCS: END task_class


class FrankaStackingFSM:
    """Select cubes and advance reactive pick-and-place phases."""

    _NEXT_PHASE = {
        PickPlacePhase.PRE_GRASP: PickPlacePhase.APPROACH,
        PickPlacePhase.APPROACH: PickPlacePhase.GRASP,
        PickPlacePhase.GRASP: PickPlacePhase.LIFT,
        PickPlacePhase.LIFT: PickPlacePhase.TRANSPORT,
        PickPlacePhase.TRANSPORT: PickPlacePhase.LOWER,
        PickPlacePhase.LOWER: PickPlacePhase.RELEASE,
        PickPlacePhase.RELEASE: PickPlacePhase.RETRACT,
    }
    _SHORT_PHASES = {
        PickPlacePhase.APPROACH,
        PickPlacePhase.LIFT,
        PickPlacePhase.TRANSPORT,
        PickPlacePhase.LOWER,
    }

    def __init__(self, config: TaskConfig) -> None:
        self._config = config
        self._active: PickPlaceState | None = None

    def reset(self) -> None:
        """Clear task state."""
        self._active = None

    # DOCS: BEGIN forward_reactive
    def forward(self, observation: Observation, *, target_reached: bool, gripper_reached: bool) -> PickPlaceDecision:
        """Advance once from an observation and controller feedback."""
        if self._active is None:
            self._active = self._select_next_cube(observation)
            if self._active is None:
                return PickPlaceDecision(done=True)

        if self._cube_was_dropped(observation):
            print(f"[drop] {self._active.color} fell out of gripper, re-decide")
            self._active = None
            return PickPlaceDecision()

        self._restart_pickup_if_cube_moved(observation)
        self._advance_if_ready(observation, target_reached, gripper_reached)
        return self._decision()

    # DOCS: END forward_reactive

    def _cube_was_dropped(self, observation: Observation) -> bool:
        active = self._active
        assert active is not None
        return active.holding and bool(
            np.linalg.norm(observation.cube_positions[active.color] - observation.end_effector_position)
            > self._config.drop_tolerance
        )

    def _restart_pickup_if_cube_moved(self, observation: Observation) -> None:
        active = self._active
        assert active is not None
        if active.holding or active.pickup_position is not None:
            return
        current_xy = observation.cube_positions[active.color][:2]
        if active.cube_reference_xy is None:
            active.cube_reference_xy = current_xy.copy()
        elif np.linalg.norm(current_xy - active.cube_reference_xy) > self._config.cube_move_tolerance:
            print(f"[resync] {active.color} moved, back to pre-grasp")
            active.phase = PickPlacePhase.PRE_GRASP
            active.phase_started = observation.time
            active.settle_started = None
            active.cube_reference_xy = current_xy.copy()

    def _select_next_cube(self, observation: Observation) -> PickPlaceState | None:
        for color in self._config.cube_order:
            if not self._at_goal(observation.cube_positions[color], self._config.stack_positions[color]):
                print(f"[decide] -> stack {color}")
                return PickPlaceState(
                    color, self._config.stack_positions[color].copy(), PickPlacePhase.PRE_GRASP, observation.time
                )
        print("[decide] -> all stacked")
        return None

    def _advance_if_ready(self, observation: Observation, target_reached: bool, gripper_reached: bool) -> None:
        active = self._active
        assert active is not None
        elapsed = max(0.0, observation.time - active.phase_started)
        if active.phase is PickPlacePhase.GRASP:
            if elapsed >= self._config.minimum_duration:
                self._advance(observation)
            return
        if active.phase is PickPlacePhase.RELEASE:
            if elapsed >= self._config.minimum_duration and gripper_reached:
                self._advance(observation)
            return

        if target_reached:
            if active.settle_started is None:
                active.settle_started = observation.time
        else:
            active.settle_started = None
        settled = (
            active.settle_started is not None
            and observation.time - active.settle_started >= self._config.settle_duration
        )
        timeout = self._config.short_phase_timeout if active.phase in self._SHORT_PHASES else self._config.phase_timeout
        if elapsed >= self._config.minimum_duration and settled:
            self._advance(observation, "settled")
        elif elapsed >= timeout:
            self._advance(observation, "timeout")

    # DOCS: BEGIN advance_task
    def _advance(self, observation: Observation, reason: str = "ok") -> None:
        active = self._active
        assert active is not None
        print(f"  [{active.color}] {active.phase.name.lower()} done ({reason})")
        if active.phase is PickPlacePhase.RETRACT:
            self._active = None
            return

        active.phase = self._NEXT_PHASE[active.phase]
        active.phase_started = observation.time
        active.settle_started = None
        if active.phase is PickPlacePhase.GRASP:
            active.pickup_position = observation.cube_positions[active.color].copy()
        if active.holding and not active.aborted and not self._stack_below_is_valid(active.color, observation):
            print(f"[abort] stack invalid below {active.color}, returning to spawn")
            active.destination = self._config.spawn_positions[active.color].copy()
            active.aborted = True

    # DOCS: END advance_task

    def _decision(self) -> PickPlaceDecision:
        if self._active is None:
            return PickPlaceDecision()
        active = self._active
        return PickPlaceDecision(
            phase=active.phase,
            color=active.color,
            destination=active.destination,
            pickup_position=active.pickup_position,
        )

    def _stack_below_is_valid(self, color: str, observation: Observation) -> bool:
        index = self._config.cube_order.index(color)
        return all(
            self._at_goal(observation.cube_positions[lower], self._config.stack_positions[lower])
            for lower in self._config.cube_order[:index]
        )

    def _at_goal(self, position: np.ndarray, goal: np.ndarray) -> bool:
        return bool(np.linalg.norm(position - goal) < self._config.goal_tolerance)


def main(args: argparse.Namespace, app: SimulationApp) -> None:
    """Run the independent scene I/O and FSM controller."""
    SimulationManager.setup_simulation(dt=_PHYSICS_DT, device="cpu")
    config = make_task_config()
    scene = FrankaStackingSceneIO(config)
    policy = FrankaStackingFSM(config)
    controller: PickPlaceController | None = None
    app.run_coroutine(scene.setup())
    app.update()

    if args.headless and not args.test:
        print("Headless: press Play in the livestream UI to begin.")
        initialized = False
    else:
        app_utils.play()
        app.update()
        scene.initialize()
        controller = make_pick_place_controller(config, policy, scene.controller_resources())
        initialized = True

    needs_reset = True
    frame_count = 0
    while app.is_running():
        app.update()
        if app_utils.is_playing() and SimulationManager.is_simulating():
            if not initialized:
                scene.initialize()
                controller = make_pick_place_controller(config, policy, scene.controller_resources())
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
