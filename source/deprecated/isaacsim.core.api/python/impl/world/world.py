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

"""Provide a comprehensive physics simulation world environment with scene management, task orchestration, and data logging capabilities."""

from __future__ import annotations

import gc

import carb
from isaacsim.core.api.loggers import DataLogger
from isaacsim.core.api.scenes.scene import Scene

# isaac-core
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.simulation_manager import IsaacEvents

# omniverse
from pxr import Usd

# python


class World(SimulationContext):
    """Provide a comprehensive physics simulation world environment with scene management and task orchestration.

    Extends SimulationContext with additional functionality for managing tasks and scenes.
    SimulationContext handles time-related events such as physics and render steps, callback
    function management for physics steps and timeline events, and stage operations.

    Includes a PhysicsContext instance for physics-related settings such as physics dt and solver type.

    Enables easy control of default reset states by adding objects to the Scene. Objects are bound
    to keywords that facilitate retrieval like a dictionary.

    Check out the required tutorials at https://docs.isaacsim.omniverse.nvidia.com/latest/index.html

    Args:
        physics_dt: dt between physics steps.
        rendering_dt: dt between rendering steps. Note: rendering means
            rendering a frame of the current application and not
            only rendering a frame to the viewports/cameras. So UI
            elements of Isaac Sim will be refreshed with this dt
            as well if running non-headless.
        stage_units_in_meters: The metric units of assets. This affects the gravity value, etc.
        physics_prim_path: Specifies the prim path to create a PhysicsScene at,
            only when no PhysicsScene is already defined.
        sim_params: Simulation parameters.
        set_defaults: Set to True to use the default settings
            [physics_dt = 1.0/ 60.0,
            stage units in meters = 1.0 (i.e. in meters),
            rendering_dt = 1.0 / 60.0,
            gravity = -9.81 m / s
            ccd_enabled,
            stabilization_enabled,
            GPU dynamics turned off,
            broadphase type is MBP,
            solver type is TGS].
        backend: Specifies the backend to be used (numpy or torch or warp).
        device: Specifies the device to be used if running on the GPU with torch or warp backends.

    Example:

    .. code-block:: python

        >>> from isaacsim.core.api import World
        >>>
        >>> world = World()
        >>> world
        <isaacsim.core.api.world.world.World object at 0x...>
    """

    _world_initialized = False
    """Class-level flag indicating whether a World instance has been created and initialized."""

    def __init__(
        self,
        physics_dt: float | None = None,
        rendering_dt: float | None = None,
        stage_units_in_meters: float | None = None,
        physics_prim_path: str = "/physicsScene",
        sim_params: dict = None,
        set_defaults: bool = True,
        backend: str = "numpy",
        device: str | None = None,
    ) -> None:
        SimulationContext.__init__(
            self,
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=stage_units_in_meters,
            physics_prim_path=physics_prim_path,
            sim_params=sim_params,
            set_defaults=set_defaults,
            backend=backend,
            device=device,
        )
        if World._world_initialized:
            carb.log_warn(
                "World is already initialized. Constructor parameters are ignored on subsequent calls. "
                "Call World.clear_instance() first to re-initialize with new parameters."
            )
            return
        World._world_initialized = True
        self._task_scene_built = False
        self._current_tasks = {}
        self._scene = Scene()
        self._data_logger = DataLogger()
        return

    """
    Instance handling.
    """

    @classmethod
    def clear_instance(cls) -> None:
        """Delete the world object, if it was instantiated before, and destroy any subscribed callback.

        Example:

        .. code-block:: python

            >>> World.clear_instance()
        """
        if World._world_initialized:
            if hasattr(SimulationContext._instance, "_scene"):
                del SimulationContext._instance._scene
                gc.collect()
            World._world_initialized = False
            SimulationContext.clear_instance()
        return

    """
    Properties.
    """

    @property
    def scene(self) -> Scene:
        """Scene instance.

        Returns:
            Scene instance.

        Example:

        .. code-block:: python

            >>> world.scene
            <isaacsim.core.api.scenes.scene.Scene object at 0x>
        """
        return self._scene

    """
    Operations - Tasks management.
    """

    def add_task(self, task: BaseTask) -> None:
        """Add a task to the task registry.

        .. note::

            Tasks should have a unique name

        Args:
            task: Task object to add.

        Raises:
            Exception: If a task with the same name already exists in the world.

        Example:

        .. code-block:: python

            >>> from isaacsim.core.api.tasks import BaseTask
            >>>
            >>> class Task(BaseTask):
            ...    def get_observations(self):
            ...        return {'obs': [0]}
            ...
            ...    def calculate_metrics(self):
            ...        return {"reward": 1}
            ...
            ...    def is_done(self):
            ...        return False
            ...
            >>> task = Task(name="custom_task")
            >>> world.add_task(task)
        """
        if task.name in self._current_tasks:
            raise Exception("Task name should be unique in the world")
        self._current_tasks[task.name] = task
        return

    def is_tasks_scene_built(self) -> bool:
        """Check if the ``set_up_scene`` method was called for each registered task.

        Returns:
            Whether the ``set_up_scene`` method was called for each registered task.

        Example:

        .. code-block:: python

            >>> # given a world instance that was rested at some point
            >>> world.is_tasks_scene_built()
            True
        """
        return self._task_scene_built

    def get_current_tasks(self) -> list[BaseTask]:
        """Get a dictionary of the registered tasks where keys are task names.

        Returns:
            Registered tasks keyed by task name.

        Example:

        .. code-block:: python

            >>> world.get_current_tasks()
            {'custom_task': <custom.task.scripts.extension.Task object at 0x...>}
        """
        return self._current_tasks

    def get_task(self, name: str) -> BaseTask:
        """Get a task by its name.

        Args:
            name: Task name to retrieve.

        Returns:
            The task with the specified name.

        Raises:
            Exception: If the task name does not exist in the current world tasks.

        Example:

        .. code-block:: python

            >>> world.get_task("custom_task")
            <custom.task.scripts.extension.Task object at 0x...>
        """
        if name not in self._current_tasks:
            raise Exception(f"task name {name} doesn't exist in the current world tasks.")
        return self._current_tasks[name]

    """
    Operations - Tasks state collection.
    """

    def get_observations(self, task_name: str | None = None) -> dict:
        """Get observations from tasks that were added.

        Args:
            task_name: Task name to ask for. If None, returns observations from all tasks.

        Returns:
            Task observations for the specified task or all tasks.

        Raises:
            Exception: If the task name does not exist in the current world tasks.

        Example:

        .. code-block:: python

            >>> world.get_observations("custom_task")
            {'obs': [0]}
        """
        if task_name is not None:
            return self.get_task(task_name).get_observations()
        else:
            observations = {}
            for task in self._current_tasks.values():
                observations.update(task.get_observations())
            return observations

    def calculate_metrics(self, task_name: str | None = None) -> dict:
        """Get metrics from tasks that were added.

        Args:
            task_name: Task name to ask for. If None, returns metrics from all tasks.

        Returns:
            Computed metrics for the specified task or all tasks.

        Raises:
            Exception: If the task name does not exist in the current world tasks.

        Example:

        .. code-block:: python

            >>> world.calculate_metrics("custom_task")
            {'reward': 1}
        """
        if task_name is not None:
            return self.get_task(task_name).calculate_metrics()
        else:
            metrics = {}
            for task in self._current_tasks.values():
                metrics.update(task.calculate_metrics())
            return metrics

    def is_done(self, task_name: str | None = None) -> bool:
        """Get the done state from tasks that were added.

        Args:
            task_name: Task name to ask for. If None, checks if all tasks are done.

        Returns:
            Whether the specified task or all tasks are done.

        Raises:
            Exception: If the task name does not exist in the current world tasks.

        Example:

        .. code-block:: python

            >>> world.is_done("custom_task")
            False
        """
        if task_name is not None:
            return self.get_task(task_name).is_done()
        else:
            result = [task.is_done() for task in self._current_tasks.values()]
            return all(result)

    """
    Operations - Data logger.
    """

    def get_data_logger(self) -> DataLogger:
        """Return the data logger of the world.

        Returns:
            Data logger instance.

        Example:

        .. code-block:: python

            >>> world.get_data_logger()
            <isaacsim.core.api.loggers.data_logger.DataLogger object at 0x...>
        """
        return self._data_logger

    """
    Operations.
    """

    def initialize_physics(self) -> None:
        """Initialize the physics simulation view and finalize each object added to the Scene.

        Example:

        .. code-block:: python

            >>> world.initialize_physics()
        """
        SimulationContext.initialize_physics(self)
        self._scene._finalize(self.physics_sim_view)
        return

    def reset(self, soft: bool = False) -> None:
        """Reset the stage to its initial state and each object included in the Scene to its default state.

        The default state is specified by the ``set_default_state`` and ``__init__`` methods.

        .. note::

            - All tasks should be added before the first reset is called unless the ``clear`` method was called.
            - All articulations should be added before the first reset is called unless the ``clear`` method was called.
            - This method takes care of initializing articulation handles with the first reset called.
            - This will do one step internally regardless.
            - Call ``post_reset`` on each object in the Scene.
            - Call ``post_reset`` on each Task.

            Things like setting PD gains should happen at a Task reset or a Robot reset since the defaults are
            restored after the ``stop`` method is called.

        .. warning::

            This method is not intended to be used in the Isaac Sim's Extensions workflow since the Omniverse Kit SDK
            application has control over the rendering steps. For the Extensions workflow, use the ``reset_async`` method
            instead.

        Args:
            soft: If set to True, simulation will not be stopped and started again. It only calls reset on the Scene
                objects.

        Example:

        .. code-block:: python

            >>> world.reset()
        """
        if not self._task_scene_built:
            for task in self._current_tasks.values():
                task.set_up_scene(self.scene)
            self._task_scene_built = True
        if not soft:
            self.stop()
        for task in self._current_tasks.values():
            task.cleanup()
        SimulationContext.reset(self, soft=soft)
        self._scene._finalize(self.physics_sim_view)
        self.scene.post_reset()
        for task in self._current_tasks.values():
            task.post_reset()

    async def reset_async_set_up_scene(self, soft: bool = False) -> None:
        """Set up the Scene for each registered task before an async reset.

        Calls ``set_up_scene`` on each Task with the World Scene.

        Args:
            soft: Unused parameter kept for compatibility with async reset methods.

        Example:

        .. code-block:: python

            >>> from omni.kit.async_engine import run_coroutine
            >>>
            >>> async def task():
            >>>     await world.reset_async_set_up_scene()
            >>>
            >>> run_coroutine(task())
        """
        for task in self._current_tasks.values():
            task.set_up_scene(self.scene)

    async def reset_async_no_set_up_scene(self, soft: bool = False) -> None:
        """Reset the stage and each object included in the Scene without calling Task ``set_up_scene``.

        The default state is specified by the ``set_default_state`` and ``__init__`` methods.

        .. note::

            - All tasks should be added before the first reset is called unless the ``clear`` method was called.
            - All articulations should be added before the first reset is called unless the ``clear`` method was called.
            - This method takes care of initializing articulation handles with the first reset called.
            - This will do one step internally regardless.
            - Call ``post_reset`` on each object in the Scene.
            - Call ``post_reset`` on each Task.

            Things like setting PD gains should happen at a Task reset or a Robot reset since the defaults are
            restored after the ``stop`` method is called.

        Args:
            soft: If set to True, simulation will not be stopped and started again. It only calls reset on the Scene
                objects.

        Example:

        .. code-block:: python

            >>> from omni.kit.async_engine import run_coroutine
            >>>
            >>> async def task():
            >>>     await world.reset_async_no_set_up_scene()
            >>>
            >>> run_coroutine(task())
        """
        if not soft:
            await self.stop_async()
        for task in self._current_tasks.values():
            task.cleanup()
        await SimulationContext.reset_async(self, soft=soft)
        self._scene._finalize(self.physics_sim_view)
        self._message_bus.dispatch_event(IsaacEvents.POST_RESET.value, payload={})
        self._scene.post_reset()
        for task in self._current_tasks.values():
            task.post_reset()
        return

    async def reset_async(self, soft: bool = False) -> None:
        """Reset the stage to its initial state and each object included in the Scene to its default state.

        The default state is specified by the ``set_default_state`` and ``__init__`` methods.

        .. note::

            - All tasks should be added before the first reset is called unless the ``clear`` method was called.
            - All articulations should be added before the first reset is called unless the ``clear`` method was called.
            - This method takes care of initializing articulation handles with the first reset called.
            - This will do one step internally regardless.
            - Call ``post_reset`` on each object in the Scene.
            - Call ``post_reset`` on each Task.

            Things like setting PD gains should happen at a Task reset or a Robot reset since the defaults are
            restored after the ``stop`` method is called.

        Args:
            soft: If set to True, simulation will not be stopped and started again. It only calls reset on the Scene
                objects.

        Example:

        .. code-block:: python

            >>> from omni.kit.async_engine import run_coroutine
            >>>
            >>> async def task():
            ...     await world.reset_async()
            ...
            >>> run_coroutine(task())
        """
        if not self._task_scene_built:
            await self.reset_async_set_up_scene()
            self._task_scene_built = True
        await self.reset_async_no_set_up_scene(soft=soft)
        return

    def step(self, render: bool = True, step_sim: bool = True, update_fabric: bool = False) -> None:
        """Step the physics simulation with or without rendering.

        .. note::

            The ``pre_step`` for each Task is called before stepping. This method also updates the Bounding Box Cache
            time for computing bounding boxes if enabled.

        .. warning::

            Calling this method with ``render`` set to True is not intended to be used in the Isaac Sim's Extensions
            workflow since the Omniverse Kit SDK application has control over the rendering steps.

        Args:
            render: Set to False to only do a physics simulation without rendering. The application UI will be frozen
                because it is not rendering in this case.
            step_sim: True to step simulation.
            update_fabric: Whether to force the update of the physics data to fabric when performing a physics-only
                step without rendering. Enable this flag to read updated data using the fabric interface after performing
                a physics-only step, such as XFormPrim's world transform.

        Raises:
            Exception: If data logging is started before adding a data frame logging function.

        Example:

        .. code-block:: python

            >>> world.step()
        """
        if self._task_scene_built:
            for task in self._current_tasks.values():
                task.pre_step(self.current_time_step_index, self.current_time)
        if self.scene._enable_bounding_box_computations:
            self.scene._bbox_cache.SetTime(Usd.TimeCode(self._current_time))

        if step_sim:
            SimulationContext.step(self, render=render, update_fabric=update_fabric)
        if self._data_logger.is_started():
            if self._data_logger._data_frame_logging_func is None:
                raise Exception("You need to add data logging function before starting the data logger")
            data = self._data_logger._data_frame_logging_func(tasks=self.get_current_tasks(), scene=self.scene)
            self._data_logger.add_data(
                data=data, current_time_step=self.current_time_step_index, current_time=self.current_time
            )
        return

    def step_async(self, step_size: float | None = None) -> None:
        """Run World pre-step updates before an external physics step.

        .. note::

            The ``pre_step`` for each Task is called before stepping. This method also updates the Bounding Box Cache
            time for computing bounding boxes if enabled.

        Args:
            step_size: Unused step size parameter.

        Raises:
            Exception: If data logging is started before adding a data frame logging function.

        Example:

        .. code-block:: python

            >>> world.step_async()
        """
        if self._task_scene_built:
            for task in self._current_tasks.values():
                task.pre_step(self.current_time_step_index, self.current_time)
        if self.scene._enable_bounding_box_computations:
            self.scene._bbox_cache.SetTime(Usd.TimeCode(self._current_time))
        if self._data_logger.is_started():
            if self._data_logger._data_frame_logging_func is None:
                raise Exception("You need to add data logging function before starting the data logger")
            data = self._data_logger._data_frame_logging_func(tasks=self.get_current_tasks(), scene=self.scene)
            self._data_logger.add_data(
                data=data, current_time_step=self.current_time_step_index, current_time=self.current_time
            )
        return

    def clear(self) -> None:
        """Clear the current stage, task registry, task scene state, and data logger, leaving the PhysicsScene and /World.

        Example:

        .. code-block:: python

            >>> world.clear()
        """
        self.scene.clear(registry_only=False)
        self._current_tasks = {}
        self._task_scene_built = False
        self._data_logger = DataLogger()
        # clear all prims in the stage.
        SimulationContext.clear(self)
