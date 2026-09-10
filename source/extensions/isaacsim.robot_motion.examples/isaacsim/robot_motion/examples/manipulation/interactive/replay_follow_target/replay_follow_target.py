"""Replay follow-target sample for replaying recorded robot trajectories and scene data."""

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

from __future__ import annotations

import json

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, GeomPrim
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.examples.base import BaseSample
from isaacsim.storage.native import get_assets_root_path


class ReplayFollowTarget(BaseSample):
    """Interactive sample that demonstrates replaying recorded robot trajectories and scene data.

    This class creates a simulation environment with a Franka robotic manipulator and provides
    trajectory-only replay for robot movements or full-scene replay with a target cube.
    """

    def __init__(self) -> None:
        super().__init__()
        self._robot: Articulation | None = None
        self._target_cube: GeomPrim | None = None
        self._target_path = "/World/TargetCube"

        self._data_frames: list[dict] = []
        self._current_time_step_index = 0
        self._physics_callback_id: int | None = None

    def setup_scene(self) -> None:
        """Set up the scene with a Franka robot and environment."""
        GroundPlane("/World/ground_plane")
        DomeLight("/World/DomeLight").set_intensities(1000)

        stage_utils.add_reference_to_stage(
            usd_path=(
                get_assets_root_path() + "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda"
            ),
            path="/World/robot",
            variants=[("Gripper", "alternatefinger"), ("Mesh", "performance")],
        )
        self._robot = Articulation("/World/robot")
        self._robot.set_default_state(dof_positions=[0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04])

    async def setup_post_load(self) -> None:
        """Called after the scene is loaded."""
        ViewportManager.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.01, 0.01, 0.01], camera="/OmniverseKit_Persp")
        self._current_time_step_index = 0

    async def setup_pre_reset(self) -> None:
        """Called before world reset."""
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None
        self._current_time_step_index = 0

    async def setup_post_reset(self) -> None:
        """Called after world reset."""
        if self._robot:
            self._robot.reset_to_default_state()
        self._current_time_step_index = 0

    async def setup_post_clear(self) -> None:
        """Called after clearing the scene."""
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None
        self._robot = None
        self._target_cube = None
        self._data_frames = []
        self._current_time_step_index = 0

    def physics_cleanup(self) -> None:
        """Clean up physics resources."""
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

    def _load_data(self, data_file: str) -> None:
        """Load replay data from a JSON file.

        Args:
            data_file: Path to the JSON data file.
        """
        with open(data_file) as f:
            content = json.load(f)
        self._data_frames = content.get("Isaac Sim Data", [])
        self._current_time_step_index = 0

    def _create_target_cube(self) -> None:
        """Create the target cube used by full-scene replay."""
        stage = stage_utils.get_current_stage(backend="usd")
        if stage.GetPrimAtPath(self._target_path).IsValid():
            self._target_cube = GeomPrim(self._target_path)
            return
        cube_shape = Cube(
            paths=self._target_path,
            positions=[0.5, 0.0, 0.3],
            sizes=0.03,
            colors="red",
        )
        self._target_cube = GeomPrim(paths=cube_shape.paths)

    def _remove_target_cube(self) -> None:
        """Remove the target cube when replaying only robot trajectories."""
        stage = stage_utils.get_current_stage(backend="usd")
        if stage.GetPrimAtPath(self._target_path).IsValid():
            stage_utils.delete_prim(self._target_path)
        self._target_cube = None

    async def _on_replay_trajectory_event_async(self, data_file: str) -> None:
        """Load and replay trajectory data.

        Args:
            data_file: Path to the trajectory data file to load and replay.
        """
        self._load_data(data_file)
        self._remove_target_cube()

        app_utils.play()
        await app_utils.update_app_async()

        self._physics_callback_id = SimulationManager.register_callback(
            self._on_replay_trajectory_step, event=SimulationEvent.PHYSICS_POST_STEP
        )

    async def _on_replay_scene_event_async(self, data_file: str) -> None:
        """Load and replay scene data (robot + target).

        Args:
            data_file: Path to the scene data file to load and replay.
        """
        self._load_data(data_file)
        self._create_target_cube()

        app_utils.play()
        await app_utils.update_app_async()

        self._physics_callback_id = SimulationManager.register_callback(
            self._on_replay_scene_step, event=SimulationEvent.PHYSICS_POST_STEP
        )

    def _on_replay_trajectory_step(self, dt: float, context: object) -> None:
        """Physics callback for replaying trajectory (robot only).

        Args:
            dt: Time delta for the physics step.
            context: Physics context for the callback.
        """
        if not self._data_frames or self._robot is None:
            return

        if self._current_time_step_index < len(self._data_frames):
            frame = self._data_frames[self._current_time_step_index]
            joint_positions = frame["data"]["applied_joint_positions"]
            self._robot.set_dof_position_targets(joint_positions)
            self._current_time_step_index += 1
        else:
            if self._physics_callback_id is not None:
                SimulationManager.deregister_callback(self._physics_callback_id)
                self._physics_callback_id = None

    def _on_replay_scene_step(self, dt: float, context: object) -> None:
        """Physics callback for replaying scene (robot + target).

        Args:
            dt: Time delta for the physics step.
            context: Physics context for the callback.
        """
        if not self._data_frames or self._robot is None or self._target_cube is None:
            return

        if self._current_time_step_index < len(self._data_frames):
            frame = self._data_frames[self._current_time_step_index]

            joint_positions = frame["data"]["applied_joint_positions"]
            self._robot.set_dof_position_targets(joint_positions)

            target_position = frame["data"]["target_position"]
            _, current_orientation = self._target_cube.get_world_poses()
            self._target_cube.set_world_poses(positions=target_position, orientations=current_orientation)

            self._current_time_step_index += 1
        else:
            if self._physics_callback_id is not None:
                SimulationManager.deregister_callback(self._physics_callback_id)
                self._physics_callback_id = None
