# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The CortexWorld extends from the core API's world object and adds the behavior portion of the.

Cortex processing pipeline.

The full Cortex processing pipeline includes:
1. Perception
2.*World modeling
3.*Logical state monitoring
4.*Behavior (decisions)
5.*Command processing (policies)
6. Control

The stared steps are included in the CortexWorld. World modeling is handled by the standard scene
representation APIs of the underlying World, and CortexWorld provides APIs for adding logical state
monitors, behaviors, and commandable robots which supply their own command APIs for supported
policies. It also provides an API for directly adding a decider network, which includes its own
logical state monitors which are automatically added.

Currently the CortexWorld only supports the standalone python app workflow.

Example usage:
    simulation_app = SimulationApp({"headless": False})
    from isaacsim.cortex.framework.robot import add_franka_to_stage
    from isaacsim.cortex.framework.cortex_world import CortexWorld

    world = CortexWorld()
    world.scene.add_default_ground_plane()

    robot = world.add_robot(add_franka_to_stage(name="franka", prim_path="/World/Franka"))

    # ...
    # Create your decider_network using the tools from df.py and dfb.py, or load it using:
    #
    # from isaacsim.cortex.framework.cortex_utils import load_behavior_module
    # decider_network = load_behavior_module(module_path).make_decider_network(robot)
    # ...

    world.add_decider_network(decider_network)

    world.run(simulation_app)
    simulation_app.close()

See standalone_examples/api/isaacsim.cortex.framework/franka_examples_main.py for details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

from isaacsim.core.api import SimulationContext, World
from isaacsim.core.prims import SingleArticulation
from isaacsim.cortex.framework.df import DfBehavior, DfLogicalState, DfNetwork
from isaacsim.cortex.framework.tools import SteadyRate


class LogicalStateMonitor:
    """A logical state monitor that can be added to the CortexWorld.

    This object interfaces with a DfLogicalState object, which owns its own monitors, to the CortexWorld.

    Args:
        name: The name used to index this logical state monitor.
        df_logical_state: The logical state object owning the underlying monitors.
    """

    def __init__(self, name: str, df_logical_state: DfLogicalState) -> None:
        self.name = name
        self.df_logical_state = df_logical_state

    def pre_step(self) -> None:
        """Process the logical state monitors of the underlying df_logical_state.

        The Cortex pipeline is processed before (pre_) stepping physics. Logical state monitors are
        stepped first, before behaviors and commanders.
        """
        for monitor in self.df_logical_state.monitors:
            monitor(self.df_logical_state)

    def post_reset(self) -> None:
        """Reset the underlying df_logical_state.

        The Cortex pipeline is reset after (post_) resetting physics. Logical state monitors are
        reset first, before behaviors and commanders.
        """
        self.df_logical_state.reset()


class Behavior:
    """A behavior that can be added to the CortexWorld.

    A behavior can be any object that implements the DfBehavior interface.

    Args:
        name: A name for this behavior used to reference the behavior.
        df_behavior: The behavior being added that implements the DfBehavior interface.
    """

    def __init__(self, name: str, df_behavior: DfBehavior) -> None:
        self.df_behavior = df_behavior
        self.name = name

    def pre_step(self) -> None:
        """Step the underlying df_behavior.

        The Cortex pipeline is processed before (pre_) stepping physics. Behaviors are stepped after
        logical state monitors, but before commanders.
        """
        self.df_behavior.step()

    def post_reset(self) -> None:
        """Reset the underlying df_behavior.

        The Cortex pipeline is reset after (post_) resetting physics. The behaviors are reset after
        logical state monitors, but before commanders.
        """
        self.df_behavior.reset()


class CommandableArticulation(ABC, SingleArticulation):
    """A commandable articulation is an articulation with a collection of commanders controlling the joints. These commanders should be stepped through a call to step_commanders()."""

    @abstractmethod
    def step_commanders(self) -> None:
        """Define how commanders are stepped each cycle.

        This method is called once per cycle.

        Raises:
            NotImplementedError: This abstract method must be implemented by deriving classes.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_commanders(self) -> None:
        """Reset each of the commanders associated with this articulation.

        Raises:
            NotImplementedError: This abstract method must be implemented by deriving classes.
        """
        raise NotImplementedError()

    def pre_step(self) -> None:
        """Step the commanders governing this commandable articulation.

        The Cortex pipeline is processed before (pre_) stepping physics. Commanders are stepped after behaviors.

        Raises:
            NotImplementedError: The commanders have not implemented stepping.
        """
        self.step_commanders()

    def post_reset(self) -> None:
        """Reset the underlying articulation and its commanders.

        The Cortex pipeline is reset after (post_) resetting physics. Commanders are reset after logical state monitors and
        behaviors, and the underlying articulation is reset before the commanders.

        Raises:
            NotImplementedError: The commanders have not implemented resetting.
        """
        super().post_reset()
        self.reset_commanders()


class CortexWorld(World):
    """The CortexWorld extends the core API's world to add the Cortex processing pipeline.

    Includes methods for adding logical state monitors, behaviors, and commandable robots. Often
    logical state monitors and behaviors come bundled in decider networks, so the CortexWorld also
    provides a convenience method for adding a decider network which both adds its logical state
    monitors and the decider network behavior.

    This class also provides a standard step() method which handles the processing of the Cortex
    pipeline as well as stopping, pausing, and playing the simulation.

    Args:
        *args: Positional arguments passed to the underlying core API World.
        **kwargs: Keyword arguments passed to the underlying core API World.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._logical_state_monitors = OrderedDict()
        self._behaviors = OrderedDict()
        self._robots = OrderedDict()

    def add_logical_state_monitor(self, logical_state_monitor: LogicalStateMonitor) -> None:
        """Adds a logical state monitor to the Cortex world.

        Multiple logical state monitors with unique names can be added. They are each stepped in the order added during the logical state monitoring phase of the Cortex pipeline.

        Args:
            logical_state_monitor: LogicalStateMonitor object representing the monitor being added.
        """
        self._logical_state_monitors[logical_state_monitor.name] = logical_state_monitor

    def add_behavior(self, behavior: Behavior) -> None:
        """Adds a behavior to the Cortex world.

        Multiple behaviors with unique names can be added. They are stepped in the order added during the behavior decisions phase of the Cortex pipeline.

        Args:
            behavior: Behavior object representing the behavior being added.
        """
        self._behaviors[behavior.name] = behavior

    def add_decider_network(self, decider_network: DfNetwork, name: Optional[str] = None) -> None:
        """Adds a decider network to the Cortex world along with any logical state monitors bundled with it.

        Args:
            decider_network: Decider network being added.
            name: Name to give the logical state monitors and decider network behavior.
                The name field can be used to add multiple decider networks with unique names
                that are stepped simultaneously.
        """
        self.add_logical_state_monitor(LogicalStateMonitor(name, decider_network.context))
        self.add_behavior(Behavior(name, decider_network))
        self.reset_cortex()

    def add_robot(self, robot: CommandableArticulation) -> CommandableArticulation:
        """Adds a commandable robot articulation to the Cortex world and scene.

        Multiple robots with unique names can be added. Their underlying commanders are stepped in the order they are added in the command API policy phase of the Cortex pipeline.

        Args:
            robot: Commandable robot being added.

        Returns:
            The added robot.
        """
        self._robots[robot.name] = robot
        self.scene.add(robot)
        return robot

    def step(self, render: bool = True, step_sim: bool = True) -> None:
        """Steps the Cortex pipeline and the underlying simulator.

        The Cortex pipeline is stepped in the order of logical state monitoring, behavior, and robot commanders. The Cortex pipeline is processed before stepping the simulator.

        Args:
            render: Whether to render this cycle.
            step_sim: Whether to step the simulation physics this cycle.

        Raises:
            Exception: If the data logger is started without a data frame logging function.
        """
        if self._task_scene_built:
            for task in self._current_tasks.values():
                task.pre_step(self.current_time_step_index, self.current_time)
            if self.is_playing():
                # Cortex pipeline: Process logical state monitors, then make decisions based on that
                # logical state (sends commands to the robot's commanders), and finally step the
                # robot's commanders to handle those commands.
                for ls_monitor in self._logical_state_monitors.values():
                    ls_monitor.pre_step()
                for behavior in self._behaviors.values():
                    behavior.pre_step()
                for robot in self._robots.values():
                    robot.pre_step()

        if self.scene._enable_bounding_box_computations:
            self.scene._bbox_cache.SetTime(Usd.TimeCode(self._current_time))

        if step_sim:
            SimulationContext.step(self, render=render)
        if self._data_logger.is_started():
            if self._data_logger._data_frame_logging_func is None:
                raise Exception("You need to add data logging function before starting the data logger")
            data = self._data_logger._data_frame_logging_func(tasks=self.get_current_tasks(), scene=self.scene)
            self._data_logger.add_data(
                data=data, current_time_step=self.current_time_step_index, current_time=self.current_time
            )
        return

    def reset(self, soft: bool = False) -> None:
        """Resets both the underlying world and the Cortex pipeline.

        The world is reset before the Cortex pipeline. See ``reset_cortex()`` for documentation on Cortex resetting.

        Args:
            soft: Whether to perform a soft reset.
        """
        super().reset(soft)
        self.reset_cortex()

    def reset_cortex(self) -> None:
        """Resets the Cortex pipeline only.

        Commanders are reset first in case logical state monitors or behaviors need to use their reset information. Logical state monitors are then reset to reset the logical state, which might be referenced by reset behaviors. Behaviors are reset last.
        """
        for robot in self._robots.values():
            robot.reset_commanders()
        for ls_monitor in self._logical_state_monitors.values():
            ls_monitor.post_reset()
        for behavior in self._behaviors.values():
            behavior.post_reset()

    def run(
        self,
        simulation_app: SimulationApp,
        render: bool = True,
        loop_fast: bool = False,
        play_on_entry: bool = False,
        is_done_cb: bool | None = None,
    ) -> None:
        """Runs the Cortex loop runner.

        This method blocks until Omniverse is exited or ``is_done_cb`` returns True. It steps everything in the world, including tasks, logical state monitors, behaviors, and robot commanders, every cycle. Cycles are run in real time at the rate given by the physics dt, usually 60 Hz. Set ``loop_fast`` to True to loop as fast as possible instead of in real time.

        Args:
            simulation_app: Simulation application handle for this Python app.
            render: Whether to render every cycle.
            loop_fast: Whether to loop as fast as possible without maintaining real time.
            play_on_entry: Whether to reset the world on entry and start the simulation playing immediately.
            is_done_cb: Function called each cycle. When it returns True, the loop exits immediately.

        Raises:
            Exception: If stepping raises because the data logger is started without a data frame logging function.
        """
        physics_dt = self.get_physics_dt()
        rate_hz = 1.0 / physics_dt
        rate = SteadyRate(rate_hz)

        if play_on_entry:
            self.reset()
            needs_reset = False  # We've already reset.
        else:
            needs_reset = True  # Reset up front the first cycle through.
        while simulation_app.is_running():
            if is_done_cb is not None and is_done_cb():
                break

            if self.is_playing():
                if needs_reset:
                    self.reset()
                    needs_reset = False
            elif self.is_stopped():
                # Every time the self steps playing we'll need to reset again when it starts again.
                needs_reset = True

            self.step(render=render)
            if not loop_fast:
                rate.sleep()
