# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared data, configuration, and scene I/O for Franka stacking demos."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Protocol

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import omni.kit.app
import warp as wp
from isaacsim.core.experimental.objects import Cone, Cube, Cylinder, DomeLight, GroundPlane, Mesh
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, RmpFlowController, load_cumotion_supported_robot
from isaacsim.storage.native import get_assets_root_path_async

_ROBOT_PATH = "/World/franka"
_CUBE_PATHS = {color: f"/World/cube_{color}" for color in ("red", "blue", "green", "yellow")}
_CUBE_COLORS = {
    "red": [0.8, 0.1, 0.1],
    "blue": [0.1, 0.1, 0.8],
    "green": [0.1, 0.7, 0.1],
    "yellow": [0.9, 0.8, 0.1],
}
_FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
_DEFAULT_JOINTS = [0.012, -0.5686, 0.0, -2.8102, 0.0, 3.0366, 0.741, 0.04, 0.04]
_DOWN_ORIENTATION = np.array([0.0, 0.0, 1.0, 0.0])


def _vector(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, kw_only=True)
class Observation:
    """One coherent estimate read from the scene."""

    sample_id: int
    time: float
    end_effector_position: np.ndarray
    gripper_position: float
    cube_positions: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "end_effector_position", _vector(self.end_effector_position))
        cubes = {name: _vector(position) for name, position in self.cube_positions.items()}
        object.__setattr__(self, "cube_positions", MappingProxyType(cubes))


class PickPlacePhase(Enum):
    PRE_GRASP = auto()
    APPROACH = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSPORT = auto()
    LOWER = auto()
    RELEASE = auto()
    RETRACT = auto()


@dataclass(frozen=True, kw_only=True)
class PickPlaceDecision:
    """Task state selected by a stacking policy."""

    phase: PickPlacePhase | None = None
    color: str | None = None
    destination: np.ndarray | None = None
    pickup_position: np.ndarray | None = None
    done: bool = False

    def __post_init__(self) -> None:
        for name in ("destination", "pickup_position"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _vector(value))


@dataclass(frozen=True, kw_only=True)
class TaskConfig:
    """Geometry and timing for the stacking task."""

    cube_order: tuple[str, ...]
    spawn_positions: Mapping[str, np.ndarray]
    stack_positions: Mapping[str, np.ndarray]
    cube_size: float = 0.05
    above_height: float = 0.10
    grasp_offset: float = -0.025
    tool_y_offset: float = -0.03
    goal_tolerance: float = 0.06
    end_effector_tolerance: float = 0.08
    gripper_tolerance: float = 0.008
    cube_move_tolerance: float = 0.05
    drop_tolerance: float = 0.10
    open_position: float = 0.04
    closed_position: float = 0.0
    minimum_duration: float = 0.5
    settle_duration: float = 0.25
    phase_timeout: float = 5.0
    short_phase_timeout: float = 2.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "spawn_positions",
            MappingProxyType({name: _vector(value) for name, value in self.spawn_positions.items()}),
        )
        object.__setattr__(
            self,
            "stack_positions",
            MappingProxyType({name: _vector(value) for name, value in self.stack_positions.items()}),
        )


def make_task_config() -> TaskConfig:
    """Create the shared Franka stacking configuration."""
    cube_size = 0.05
    cube_order = ("red", "blue", "green", "yellow")
    spawn_positions = {
        "red": np.array([0.45, -0.30, cube_size / 2]),
        "blue": np.array([0.55, -0.30, cube_size / 2]),
        "green": np.array([0.45, -0.45, cube_size / 2]),
        "yellow": np.array([0.55, -0.45, cube_size / 2]),
    }
    stack_positions = {
        color: np.array([0.55, 0.20, cube_size / 2 + index * cube_size]) for index, color in enumerate(cube_order)
    }
    return TaskConfig(
        cube_order=cube_order,
        spawn_positions=spawn_positions,
        stack_positions=stack_positions,
        cube_size=cube_size,
    )


class _PickPlacePolicy(Protocol):
    def reset(self) -> None: ...

    def forward(
        self, observation: Observation, *, target_reached: bool, gripper_reached: bool
    ) -> PickPlaceDecision: ...


# DOCS: BEGIN gripper_controller
class GripperController(mg.BaseController):
    """Command the Franka finger joints to one fixed position."""

    def __init__(self, *, joint_space: list[str], finger_joint_names: tuple[str, ...], target_position: float) -> None:
        self._joint_space = joint_space
        self._finger_names = [name for name in finger_joint_names if name in joint_space]
        self._positions = wp.array([target_position] * len(self._finger_names), dtype=wp.float32)

    def reset(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> bool:
        return True

    def forward(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> mg.RobotState:
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=self._joint_space,
                positions=(self._finger_names, self._positions),
            )
        )


# DOCS: END gripper_controller


class PickPlaceController(mg.BaseController):
    """Run a stacking policy through arm and gripper controllers."""

    _CLOSED_PHASES = {
        PickPlacePhase.GRASP,
        PickPlacePhase.LIFT,
        PickPlacePhase.TRANSPORT,
        PickPlacePhase.LOWER,
    }

    def __init__(
        self,
        *,
        config: TaskConfig,
        policy: _PickPlacePolicy,
        arm_controller: mg.BaseController,
        joint_space: list[str],
        site_space: list[str],
        tool_frame: str,
    ) -> None:
        self._config = config
        self._policy = policy
        self._site_space = site_space
        self._tool_frame = tool_frame
        # DOCS: BEGIN combined_controller
        # Pair the arm controller with an open or closed gripper. Each pair is a
        # single controller that moves the arm and holds the hand at one width.
        open_gripper = GripperController(
            joint_space=joint_space,
            finger_joint_names=_FINGER_JOINTS,
            target_position=config.open_position,
        )
        closed_gripper = GripperController(
            joint_space=joint_space,
            finger_joint_names=_FINGER_JOINTS,
            target_position=config.closed_position,
        )
        open_and_move = mg.CombinedController([arm_controller, open_gripper])
        close_and_move = mg.CombinedController([arm_controller, closed_gripper])
        # DOCS: END combined_controller
        # DOCS: BEGIN selectable_controller
        # Map every phase to the pair it needs. The container swaps between them
        # by enum key and resets each controller the first time it becomes active.
        self._controllers = mg.SelectableController(
            controller_options={
                PickPlacePhase.PRE_GRASP: open_and_move,
                PickPlacePhase.APPROACH: open_and_move,
                PickPlacePhase.GRASP: close_and_move,
                PickPlacePhase.LIFT: close_and_move,
                PickPlacePhase.TRANSPORT: close_and_move,
                PickPlacePhase.LOWER: close_and_move,
                PickPlacePhase.RELEASE: open_and_move,
                PickPlacePhase.RETRACT: open_and_move,
            },
            initial_controller_selection=PickPlacePhase.PRE_GRASP,
        )
        # DOCS: END selectable_controller
        self._target: np.ndarray | None = None
        self._gripper_open: bool | None = None
        self.done = False

    def reset(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> bool:
        self._policy.reset()
        self._target = None
        self._gripper_open = None
        self.done = False
        return self._controllers.reset(estimated_state, setpoint_state, t, **kwargs)

    def forward(
        self,
        estimated_state: mg.RobotState,
        setpoint_state: mg.RobotState | None,
        t: float,
        *,
        observation: Observation,
        **kwargs: object,
    ) -> mg.RobotState | None:
        decision = self._policy.forward(
            observation,
            target_reached=self._target_reached(observation),
            gripper_reached=self._gripper_reached(observation),
        )
        self.done = decision.done
        phase = decision.phase if decision.phase is not None else PickPlacePhase.RELEASE
        self._target = self._target_for_decision(observation, decision)
        self._gripper_open = phase not in self._CLOSED_PHASES
        # DOCS: BEGIN phase_to_controller
        # The policy chose a phase; the container maps it to the matching
        # controller and runs it. This is the only line that couples the
        # task decision to the low-level controllers.
        self._controllers.set_next_controller(phase)
        setpoint = None if self._target is None else self._setpoint(self._target)
        return self._controllers.forward(estimated_state, setpoint, t, **kwargs)
        # DOCS: END phase_to_controller

    def _target_reached(self, observation: Observation) -> bool:
        return self._target is not None and bool(
            np.linalg.norm(observation.end_effector_position - self._target) < self._config.end_effector_tolerance
        )

    def _gripper_reached(self, observation: Observation) -> bool:
        gripper_target = None
        if self._gripper_open is not None:
            gripper_target = self._config.open_position if self._gripper_open else self._config.closed_position
        return (
            gripper_target is not None
            and abs(observation.gripper_position - gripper_target) < self._config.gripper_tolerance
        )

    def _target_for_decision(self, observation: Observation, decision: PickPlaceDecision) -> np.ndarray | None:
        if decision.phase is None:
            return None
        if decision.color is None or decision.destination is None:
            raise ValueError("Active pick-place decisions require a color and destination")
        source = decision.pickup_position
        if source is None:
            source = observation.cube_positions[decision.color]
        high = self._config.above_height
        low = self._config.cube_size / 2 + self._config.grasp_offset
        if decision.phase in {PickPlacePhase.PRE_GRASP, PickPlacePhase.LIFT}:
            target = np.array([source[0], source[1], source[2] + high])
        elif decision.phase in {PickPlacePhase.APPROACH, PickPlacePhase.GRASP}:
            target = np.array([source[0], source[1], source[2] + low])
        elif decision.phase in {PickPlacePhase.TRANSPORT, PickPlacePhase.RETRACT}:
            target = np.array([decision.destination[0], decision.destination[1], decision.destination[2] + high])
        else:
            target = np.array([decision.destination[0], decision.destination[1], decision.destination[2] + low])
        target[1] += self._config.tool_y_offset
        return target

    def _setpoint(self, position: np.ndarray) -> mg.RobotState:
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=self._site_space,
                positions=([self._tool_frame], wp.array([position.tolist()], dtype=wp.float32)),
                orientations=([self._tool_frame], wp.array([_DOWN_ORIENTATION.tolist()], dtype=wp.float32)),
            )
        )


@dataclass(frozen=True, kw_only=True)
class ControllerResources:
    arm_controller: mg.BaseController
    joint_space: list[str]
    site_space: list[str]
    tool_frame: str


def make_pick_place_controller(
    config: TaskConfig, policy: _PickPlacePolicy, resources: ControllerResources
) -> PickPlaceController:
    """Compose a policy with initialized robot controllers and metadata."""
    return PickPlaceController(
        config=config,
        policy=policy,
        arm_controller=resources.arm_controller,
        joint_space=resources.joint_space,
        site_space=resources.site_space,
        tool_frame=resources.tool_frame,
    )


class FrankaStackingSceneIO:
    """Read scene estimates and apply scenario-independent commands."""

    def __init__(self, config: TaskConfig) -> None:
        self._config = config
        self._articulation: Articulation | None = None
        self._end_effector: GeomPrim | None = None
        self._cubes: dict[str, GeomPrim] = {}
        self._arm_controller: RmpFlowController | None = None
        self._world: mg.WorldBinding | None = None
        self._joint_space: list[str] = []
        self._site_space: list[str] = []
        self._tool_frame = ""
        self._finger_indices: list[int] = []
        self._sample_id = -1
        self._estimated: mg.RobotState | None = None
        self._observation: Observation | None = None

    async def setup(self) -> None:
        """Create the robot, cubes, ground, light, and camera."""
        assets_root = await get_assets_root_path_async()
        stage_utils.add_reference_to_stage(
            usd_path=assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            path=_ROBOT_PATH,
            variants=[("Gripper", "AlternateFinger"), ("Mesh", "Performance")],
        )
        GroundPlane("/World/GroundPlane")
        DomeLight("/World/DomeLight").set_intensities(1000)
        for color in self._config.cube_order:
            path = _CUBE_PATHS[color]
            Cube(
                paths=path,
                positions=self._config.spawn_positions[color],
                sizes=self._config.cube_size,
                colors=_CUBE_COLORS[color],
            )
            RigidPrim(paths=path)
            GeomPrim(paths=path, apply_collision_apis=True)

        await omni.kit.app.get_app().next_update_async()
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[1.5, 0.5, 1.2], target=[0.5, 0.0, 0.2])
        self._articulation = Articulation(_ROBOT_PATH)
        self._articulation.set_default_state(dof_positions=[_DEFAULT_JOINTS])
        self._cubes = {color: GeomPrim(paths=_CUBE_PATHS[color]) for color in self._config.cube_order}
        await omni.kit.app.get_app().next_update_async()

    def initialize(self) -> None:
        """Construct the motion backend after simulation starts."""
        articulation = self._require_articulation()
        robot_config = load_cumotion_supported_robot("franka")
        robot_position, robot_orientation = articulation.get_world_poses()
        objects = mg.SceneQuery().get_prims_in_aabb(
            search_box_origin=robot_position.numpy()[0],
            search_box_minimum=[-10.0, -10.0, -10.0],
            search_box_maximum=[10.0, 10.0, 10.0],
            tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
            exclude_prim_paths=[_ROBOT_PATH, *_CUBE_PATHS.values()],
        )
        obstacle_strategy = mg.ObstacleStrategy()
        for prim_type in (Mesh, Cone, Cylinder):
            obstacle_strategy.set_default_configuration(prim_type, mg.ObstacleConfiguration("obb", 0.01))
        self._world = mg.WorldBinding(
            world_interface=CumotionWorldInterface(),
            obstacle_strategy=obstacle_strategy,
            tracked_prims=objects,
            tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
        )
        self._world.initialize()
        self._world.get_world_interface().update_world_to_robot_root_transforms((robot_position, robot_orientation))
        self._world.synchronize_transforms()

        self._joint_space = articulation.dof_names
        self._site_space = robot_config.robot_description.tool_frame_names()
        self._tool_frame = self._site_space[0]
        self._finger_indices = [self._joint_space.index(name) for name in _FINGER_JOINTS if name in self._joint_space]
        self._arm_controller = RmpFlowController(
            cumotion_robot=robot_config,
            cumotion_world_interface=self._world.get_world_interface(),
            robot_joint_space=self._joint_space,
            robot_site_space=self._site_space,
            tool_frame=self._tool_frame,
        )
        self._arm_controller.get_rmp_flow_config().set_param("cspace_target_rmp/metric_scalar", 50.0)

        link_names = articulation.link_names
        end_effector = "panda_leftfinger" if "panda_leftfinger" in link_names else "panda_hand"
        self._end_effector = GeomPrim(paths=articulation.link_paths[0][link_names.index(end_effector)])
        self.reset()

    def reset(self) -> None:
        """Reset the articulation and read/apply state."""
        self._require_articulation().reset_to_default_state()
        self._sample_id = -1
        self._estimated = None
        self._observation = None

    def controller_resources(self) -> ControllerResources:
        """Return initialized motion-controller resources."""
        return ControllerResources(
            arm_controller=self._require_arm_controller(),
            joint_space=self._joint_space.copy(),
            site_space=self._site_space.copy(),
            tool_frame=self._tool_frame,
        )

    def read(self, time: float) -> tuple[mg.RobotState, Observation]:
        """Capture one coherent observation and its backend robot state."""
        articulation = self._require_articulation()
        if self._end_effector is None:
            raise RuntimeError("Call initialize() before read()")
        if self._observation is not None:
            raise RuntimeError("Apply the previous observation's command before reading again")
        self._synchronize_world()
        self._sample_id += 1
        self._estimated = self._estimated_state()
        positions = articulation.get_dof_positions().numpy().flatten()
        gripper_position = (
            float(np.mean([positions[index] for index in self._finger_indices])) if self._finger_indices else 0.0
        )
        self._observation = Observation(
            sample_id=self._sample_id,
            time=time,
            end_effector_position=self._end_effector.get_world_poses()[0].numpy()[0],
            gripper_position=gripper_position,
            cube_positions={color: cube.get_world_poses()[0].numpy()[0] for color, cube in self._cubes.items()},
        )
        return self._estimated, self._observation

    def apply(self, desired: mg.RobotState | None) -> None:
        """Apply controller output for the latest observation."""
        if self._observation is None or self._estimated is None:
            raise RuntimeError("Call read() before apply()")
        self._apply_joints(desired)
        self._estimated = None
        self._observation = None

    def _synchronize_world(self) -> None:
        world = self._require_world()
        world.get_world_interface().update_world_to_robot_root_transforms(
            self._require_articulation().get_world_poses()
        )
        world.synchronize_transforms()

    def _estimated_state(self) -> mg.RobotState:
        articulation = self._require_articulation()
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=self._joint_space,
                positions=(self._joint_space, articulation.get_dof_positions()),
                velocities=(self._joint_space, articulation.get_dof_velocities()),
            )
        )

    def _apply_joints(self, desired: mg.RobotState | None) -> None:
        if desired is not None and desired.joints is not None and desired.joints.positions is not None:
            self._require_articulation().set_dof_position_targets(
                desired.joints.positions, dof_indices=desired.joints.position_indices
            )

    def _require_articulation(self) -> Articulation:
        if self._articulation is None:
            raise RuntimeError("Call setup() before using SceneIO")
        return self._articulation

    def _require_arm_controller(self) -> RmpFlowController:
        if self._arm_controller is None:
            raise RuntimeError("Call initialize() before using SceneIO")
        return self._arm_controller

    def _require_world(self) -> mg.WorldBinding:
        if self._world is None:
            raise RuntimeError("Call initialize() before using SceneIO")
        return self._world
