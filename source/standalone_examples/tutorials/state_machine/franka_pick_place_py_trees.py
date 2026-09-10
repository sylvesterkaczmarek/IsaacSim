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

"""Franka reactive stacking with a py_trees behavior tree."""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
import py_trees
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
S = py_trees.common.Status


@dataclass
class _Context:
    config: TaskConfig
    observation: Observation | None = None
    target_reached: bool = False
    gripper_reached: bool = False
    decision: PickPlaceDecision = PickPlaceDecision()


def _at_goal(position: np.ndarray, goal: np.ndarray, config: TaskConfig) -> bool:
    return bool(np.linalg.norm(position - goal) < config.goal_tolerance)


# DOCS: BEGIN behaviour_primitives
class Phase(py_trees.behaviour.Behaviour):
    """Run one phase using the current observation and prior-command feedback."""

    def __init__(self, context: _Context, phase: PickPlacePhase, *, spawn: bool = False, drift: bool = False) -> None:
        super().__init__(phase.name.title())
        self.context = context
        self.phase = phase
        self.spawn = spawn
        self.drift = drift
        self.bb = py_trees.blackboard.Client(name=self.name)
        for key in ("pick_target", "holding", "pickup_position"):
            self.bb.register_key(key, access=py_trees.common.Access.READ)
            self.bb.register_key(key, access=py_trees.common.Access.WRITE)

    def initialise(self) -> None:
        """Reset phase timing."""
        self._started = self.context.observation.time
        self._settle_started: float | None = None
        self._anchor_xy: np.ndarray | None = None
        self._first_tick = True
        if self.phase is PickPlacePhase.GRASP:
            color = self.bb.pick_target
            self.bb.pickup_position = self.context.observation.cube_positions[color].copy()

    def update(self) -> py_trees.common.Status:
        """Emit the phase decision and report progress."""
        observation = self.context.observation
        color = self.bb.holding if self.spawn else self.bb.pick_target
        self._publish_decision(color)
        if self._pickup_target_drifted(observation, color):
            print(f"  [{self.name}] target drifted, restarting pickup")
            return S.FAILURE
        return S.SUCCESS if self._phase_finished(observation) else S.RUNNING

    def _publish_decision(self, color: str) -> None:
        destination = (
            self.context.config.spawn_positions[color] if self.spawn else self.context.config.stack_positions[color]
        )
        pickup = self.bb.pickup_position if self.bb.pickup_position is not None else None
        self.context.decision = PickPlaceDecision(
            phase=self.phase,
            color=color,
            destination=destination,
            pickup_position=pickup,
        )

    def _pickup_target_drifted(self, observation: Observation, color: str) -> bool:
        if not self.drift:
            return False
        current_xy = observation.cube_positions[color][:2]
        if self._anchor_xy is None:
            self._anchor_xy = current_xy.copy()
            return False
        return bool(np.linalg.norm(current_xy - self._anchor_xy) > self.context.config.cube_move_tolerance)

    def _phase_finished(self, observation: Observation) -> bool:
        elapsed = max(0.0, observation.time - self._started)
        if self.phase is PickPlacePhase.GRASP:
            return elapsed >= self.context.config.minimum_duration
        if self.phase is PickPlacePhase.RELEASE:
            return elapsed >= self.context.config.minimum_duration and self.context.gripper_reached

        if not self._first_tick and self.context.target_reached:
            if self._settle_started is None:
                self._settle_started = observation.time
        else:
            self._settle_started = None
        self._first_tick = False
        settled = (
            self._settle_started is not None
            and observation.time - self._settle_started >= self.context.config.settle_duration
        )
        short = self.phase in {
            PickPlacePhase.APPROACH,
            PickPlacePhase.LIFT,
            PickPlacePhase.TRANSPORT,
            PickPlacePhase.LOWER,
        }
        timeout = self.context.config.short_phase_timeout if short else self.context.config.phase_timeout
        if elapsed >= self.context.config.minimum_duration and settled:
            return True
        if elapsed >= timeout:
            print(f"  [{self.name}] timeout, advancing")
            return True
        return False


class Action(py_trees.behaviour.Behaviour):
    """Run one blackboard action."""

    def __init__(self, name: str, action: Callable[[], None]) -> None:
        super().__init__(name)
        self._action = action

    def update(self) -> py_trees.common.Status:
        """Run the action."""
        self._action()
        return S.SUCCESS


# DOCS: END behaviour_primitives


class AllStacked(py_trees.behaviour.Behaviour):
    """Succeed when every cube is at its goal."""

    def __init__(self, context: _Context, bb: py_trees.blackboard.Client) -> None:
        super().__init__("AllStacked?")
        self.context, self.bb = context, bb

    def update(self) -> py_trees.common.Status:
        """Check task completion."""
        observation = self.context.observation
        done = self.bb.holding is None and all(
            _at_goal(observation.cube_positions[color], self.context.config.stack_positions[color], self.context.config)
            for color in self.context.config.cube_order
        )
        if done:
            self.context.decision = PickPlaceDecision(done=True)
        return S.SUCCESS if done else S.FAILURE


class SelectNextCube(py_trees.behaviour.Behaviour):
    """Select the first cube not at its goal."""

    def __init__(self, context: _Context, bb: py_trees.blackboard.Client) -> None:
        super().__init__("SelectNextCube")
        self.context, self.bb = context, bb

    def update(self) -> py_trees.common.Status:
        """Write the next cube to the blackboard."""
        observation = self.context.observation
        for color in self.context.config.cube_order:
            if not _at_goal(
                observation.cube_positions[color], self.context.config.stack_positions[color], self.context.config
            ):
                self.bb.pick_target = color
                self.bb.pickup_position = None
                print(f"[select] -> {color}")
                return S.SUCCESS
        return S.FAILURE


class CubeDropped(py_trees.behaviour.Behaviour):
    """Detect a held cube outside the gripper."""

    def __init__(self, context: _Context, bb: py_trees.blackboard.Client) -> None:
        super().__init__("CubeDropped?")
        self.context, self.bb = context, bb

    def update(self) -> py_trees.common.Status:
        """Check the held cube distance."""
        color = self.bb.holding
        if color is None:
            return S.FAILURE
        observation = self.context.observation
        dropped = np.linalg.norm(observation.cube_positions[color] - observation.end_effector_position)
        return S.SUCCESS if dropped > self.context.config.drop_tolerance else S.FAILURE


class StackBelowBroken(py_trees.behaviour.Behaviour):
    """Detect a displaced cube below the held cube."""

    def __init__(self, context: _Context, bb: py_trees.blackboard.Client) -> None:
        super().__init__("StackBelowBroken?")
        self.context, self.bb = context, bb

    def update(self) -> py_trees.common.Status:
        """Check lower stack goals."""
        color = self.bb.holding
        if color is None:
            return S.FAILURE
        observation = self.context.observation
        index = self.context.config.cube_order.index(color)
        broken = any(
            not _at_goal(
                observation.cube_positions[below], self.context.config.stack_positions[below], self.context.config
            )
            for below in self.context.config.cube_order[:index]
        )
        return S.SUCCESS if broken else S.FAILURE


# DOCS: BEGIN build_tree
def build_tree(context: _Context) -> py_trees.trees.BehaviourTree:
    """Build the reactive stacking tree."""
    bb = py_trees.blackboard.Client(name="stacking")
    for key in ("pick_target", "holding", "pickup_position"):
        bb.register_key(key, access=py_trees.common.Access.READ)
        bb.register_key(key, access=py_trees.common.Access.WRITE)
    bb.pick_target, bb.holding, bb.pickup_position = "", None, None

    phase = lambda value, **kwargs: Phase(context, value, **kwargs)
    action = lambda name, fn: Action(name, fn)

    def clear_dropped() -> None:
        bb.holding = None
        context.decision = PickPlaceDecision()

    pickup = py_trees.composites.Sequence(
        "Pickup",
        memory=True,
        children=[phase(PickPlacePhase.PRE_GRASP), phase(PickPlacePhase.APPROACH, drift=True)],
    )
    place = py_trees.composites.Sequence(
        "Place",
        memory=True,
        children=[
            phase(PickPlacePhase.LIFT),
            phase(PickPlacePhase.TRANSPORT),
            phase(PickPlacePhase.LOWER),
            phase(PickPlacePhase.RELEASE),
            phase(PickPlacePhase.RETRACT),
            action("ClearHolding", lambda: setattr(bb, "holding", None)),
        ],
    )
    normal = py_trees.composites.Sequence(
        "Normal",
        memory=True,
        children=[
            SelectNextCube(context, bb),
            pickup,
            phase(PickPlacePhase.GRASP),
            action("MarkHolding", lambda: setattr(bb, "holding", bb.pick_target)),
            place,
        ],
    )
    drop_recovery = py_trees.composites.Sequence(
        "DropRecovery",
        memory=True,
        children=[
            CubeDropped(context, bb),
            action("ClearDropped", clear_dropped),
        ],
    )
    stack_recovery = py_trees.composites.Sequence(
        "StackRecovery",
        memory=True,
        children=[
            StackBelowBroken(context, bb),
            phase(PickPlacePhase.LIFT, spawn=True),
            phase(PickPlacePhase.TRANSPORT, spawn=True),
            phase(PickPlacePhase.LOWER, spawn=True),
            phase(PickPlacePhase.RELEASE, spawn=True),
            phase(PickPlacePhase.RETRACT, spawn=True),
            action("ClearHolding", lambda: setattr(bb, "holding", None)),
        ],
    )
    root = py_trees.composites.Selector(
        "Root",
        memory=False,
        children=[AllStacked(context, bb), drop_recovery, stack_recovery, normal],
    )
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup()
    return tree


# DOCS: END build_tree


class FrankaStackingTree:
    """Expose the behavior tree through the shared policy interface."""

    def __init__(self, config: TaskConfig) -> None:
        self._context = _Context(config)
        self.tree = build_tree(self._context)

    def reset(self) -> None:
        """Rebuild the tree and clear its task state."""
        self.tree = build_tree(self._context)

    def forward(self, observation: Observation, *, target_reached: bool, gripper_reached: bool) -> PickPlaceDecision:
        """Tick the tree from one observation and controller feedback."""
        self._context.observation = observation
        self._context.target_reached = target_reached
        self._context.gripper_reached = gripper_reached
        self._context.decision = PickPlaceDecision()
        self.tree.tick()
        return self._context.decision


def main(args: argparse.Namespace, app: SimulationApp) -> None:
    """Run the behavior-tree stacking example."""
    SimulationManager.setup_simulation(dt=_PHYSICS_DT, device="cpu")
    config = make_task_config()
    scene = FrankaStackingSceneIO(config)
    policy = FrankaStackingTree(config)
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
            if frame_count % 60 == 0:
                print(py_trees.display.unicode_tree(root=policy.tree.root, show_status=True))
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
