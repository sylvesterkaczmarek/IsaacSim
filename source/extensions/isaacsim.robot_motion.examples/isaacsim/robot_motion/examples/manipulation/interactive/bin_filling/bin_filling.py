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

"""Interactive bin filling example with UR10 robot."""

from __future__ import annotations

import random

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
from isaacsim.core.experimental.utils.transform import (
    euler_angles_to_quaternion,
    quaternion_conjugate,
    quaternion_multiplication,
)
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.examples.base import BaseSample
from isaacsim.robot.surface_gripper import _surface_gripper as surface_gripper
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, RmpFlowController, load_cumotion_supported_robot
from isaacsim.robot_motion.examples.manipulation import get_robot_config
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf

ROBOT_PRIM_PATH = "/World/Scene/ur10"
BIN_PRIM_PATH = "/World/Scene/bin"

PIPE_POSITION = [0.0, 0.85, 1.2]
TARGET_POSITION = [0.0, 0.85, -0.44]
EE_OFFSET = [0.0, -0.15, 0.03]
SUCTION_TIP_LOCAL_OFFSET = [0.161709, 0.0, 0.0]

CUBE_SIZE = 0.05
MAX_CUBES = 50

EVENTS_DT = [0.01, 0.0035, 0.01, 1.0, 0.008, 0.005, 0.005]
INITIAL_EE_HEIGHT = 0.3


def _rotate_vector(orientation: tuple[float, float, float, float] | np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(orientation)
    w, u = q[0], q[1:]
    return vector + 2.0 * w * np.cross(u, vector) + 2.0 * np.cross(u, np.cross(u, vector))


def _add_reference(
    usd_path: str,
    prim_path: str,
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    variants: list[tuple[str, str]] | None = None,
) -> None:
    stage_utils.add_reference_to_stage(usd_path=usd_path, path=prim_path, variants=variants)
    XformPrim(
        prim_path,
        translations=[translation],
        orientations=[orientation],
        scales=[scale],
        reset_xform_op_properties=True,
    )


def _build_scene(assets_root: str) -> None:
    """Compose the bin-filling workcell from maintained leaf assets."""
    stage_utils.define_prim("/World/Scene")
    robot_config = get_robot_config("ur10")
    _add_reference(
        assets_root + robot_config.asset_path,
        ROBOT_PRIM_PATH,
        variants=list(robot_config.variants),
    )
    _add_reference(
        assets_root + "/Isaac/Props/KLT_Bin/small_KLT.usd",
        BIN_PRIM_PATH,
        translation=(0.0, 0.8152485474516027, -0.43576),
    )
    bin_prim = stage_utils.get_current_stage(backend="usd").GetPrimAtPath(BIN_PRIM_PATH)
    for name, value in (
        ("physxRigidBody:maxDepenetrationVelocity", 0.0005),
        ("physxRigidBody:sleepThreshold", 0.002),
        ("physxRigidBody:stabilizationThreshold", 1.0e-8),
    ):
        bin_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
    XformPrim(
        [f"{BIN_PRIM_PATH}/Visuals", f"{BIN_PRIM_PATH}/Collision"],
        scales=[(0.992, 0.992, 1.073)],
        reset_xform_op_properties=True,
    )
    _add_reference(assets_root + "/Isaac/Props/Mounts/ur10_mount.usd", "/World/Scene/ur10_mount")
    stage_utils.define_prim("/World/Scene/pipe")
    XformPrim(
        "/World/Scene/pipe",
        translations=[(-0.1, 0.81, 1.15)],
        orientations=[(0.7071068, 0.0, 0.0, -0.7071068)],
        scales=[(1.5, 1.5, 1.5)],
        reset_xform_op_properties=True,
    )
    _add_reference(
        assets_root + "/Isaac/Props/PipeFunnel/pipe_funnel.usd",
        "/World/Scene/pipe/pipe_funnel",
        orientation=(0.5, 0.5, 0.5, 0.5),
        scale=(10.0, 10.0, 10.0),
    )
    XformPrim(
        "/World/Scene/pipe/pipe_funnel/Cylinder_004_Cylinder_008",
        orientations=[(0.0, 1.0, 0.0, 0.0)],
        scales=[(0.01, 0.01, 0.01)],
        reset_xform_op_properties=True,
    )
    _add_reference(
        assets_root + "/Isaac/Environments/Grid/gridroom_curved.usd",
        "/World/Scene/gridroom_curved",
        translation=(0.512, -0.908, -0.5095),
    )
    XformPrim(
        f"{BIN_PRIM_PATH}/Collision/Cube_03",
        translations=[(0.0, 0.0, -0.0643603558184001)],
        scales=[(0.1703629, 0.26209676, 0.02)],
        reset_xform_op_properties=True,
    )


class BinFilling(BaseSample):
    """Interactive bin filling example using a UR10 robot with experimental APIs.

    A UR10 robot picks up a bin and moves it under a pipe where cubes drop from
    above. The example uses a 7-phase pick-and-hold state machine with RMPFlow-based
    end-effector control and dynamic cube spawning.

    Phase summary:
        0: Move EE above bin at initial height.
        1: Lower EE to bin height.
        2: Wait for inertia to settle.
        3: Close gripper (grasp bin).
        4: Lift EE back to initial height.
        5: Move EE to placing position.
        6: Pause, trigger cube drop, and hold bin indefinitely.
    """

    def __init__(self) -> None:
        super().__init__()
        self._robot: Articulation | None = None
        self._controller: RmpFlowController | None = None
        self._cumotion_robot = None
        self._world_binding: mg.WorldBinding | None = None
        self._tool_frame: str | None = None
        self._site_space: list[str] | None = None
        self._bin_prim: RigidPrim | None = None
        self._physics_callback_id: int | None = None

        self._cube_pool: RigidPrim | None = None
        self._active_cubes = 0
        self._cubes_to_add = 0

        self._event = 0
        self._t = 0.0
        self._pick_position: np.ndarray | None = None
        self._pick_height = 0.0
        self._pause = False
        self._added_cubes = False
        self._step_index = 0
        self._rmpflow_reset_needed = True
        self._motion_generation_t = 0.0

        self._ee_orientation: np.ndarray | None = None
        self._tool_orientation: np.ndarray | None = None
        self._suction_offset_world: np.ndarray | None = None
        self._surface_gripper_path = f"{ROBOT_PRIM_PATH}/ee_link/SurfaceGripper"
        self._surface_gripper_iface = surface_gripper.acquire_surface_gripper_interface()

    # ------------------------------------------------------------------
    # Scene lifecycle
    # ------------------------------------------------------------------

    def setup_scene(self) -> None:
        """Build the scene, wrap the robot and bin, and pre-allocate the cube pool."""
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Could not find Isaac Sim assets folder.")
        _build_scene(assets_root)

        self._robot = Articulation(ROBOT_PRIM_PATH)
        self._robot.set_default_state(dof_positions=[-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])

        self._bin_prim = RigidPrim(BIN_PRIM_PATH)

        self._create_cube_pool()

    async def setup_post_load(self) -> None:
        """Set the camera, pre-compute the EE orientation, and create RMPflow state."""
        ViewportManager.set_camera_view(eye=[2.5, 1.8, 1.2], target=[0.0, 0.3, 0.3], camera="/OmniverseKit_Persp")
        self._ee_orientation = euler_angles_to_quaternion([np.pi, 0.0, -np.pi / 2.0], extrinsic=False).numpy()
        self._tool_orientation = self._get_tool_orientation_from_ee_orientation(self._ee_orientation)
        self._suction_offset_world = _rotate_vector(self._ee_orientation, np.asarray(SUCTION_TIP_LOCAL_OFFSET))
        self._build_world_binding()
        self._build_controller()

    async def setup_pre_reset(self) -> None:
        """Deregister physics callback, deactivate cubes, and reset state machine."""
        self._remove_physics_callback()
        self._deactivate_all_cubes()
        self._reset_state_machine()

    async def setup_post_reset(self) -> None:
        """Reset the robot to its default pose."""
        if self._robot is not None:
            self._robot.reset_to_default_state()

    async def setup_post_clear(self) -> None:
        """Release all references when the scene is cleared."""
        self._cleanup_all()

    def physics_cleanup(self) -> None:
        """Release all references on extension hot-reload."""
        self._cleanup_all()

    def _cleanup_all(self) -> None:
        """Release all internal references and reset state."""
        self._remove_physics_callback()
        self._robot = None
        self._controller = None
        self._cumotion_robot = None
        self._world_binding = None
        self._tool_frame = None
        self._site_space = None
        self._bin_prim = None
        self._cube_pool = None
        self._active_cubes = 0
        self._cubes_to_add = 0
        self._ee_orientation = None
        self._tool_orientation = None
        self._suction_offset_world = None
        self._reset_state_machine()

    def _reset_state_machine(self) -> None:
        """Reset the pick-place state machine to its initial state."""
        self._event = 0
        self._t = 0.0
        self._pick_position = None
        self._pick_height = 0.0
        self._pause = False
        self._added_cubes = False
        self._step_index = 0
        self._rmpflow_reset_needed = True
        self._motion_generation_t = 0.0

    # ------------------------------------------------------------------
    # Physics callback management
    # ------------------------------------------------------------------

    def _remove_physics_callback(self) -> None:
        """Deregister the active physics callback if one exists."""
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

    # ------------------------------------------------------------------
    # RMPflow controller management
    # ------------------------------------------------------------------

    def _build_world_binding(self) -> None:
        """Create an empty cuMotion world binding matching the legacy RMPFlow example."""
        if self._robot is None:
            return

        robot_pos, robot_ori = self._robot.get_world_poses()
        self._world_binding = mg.WorldBinding(
            world_interface=CumotionWorldInterface(),
            obstacle_strategy=mg.ObstacleStrategy(),
            tracked_prims=[],
            tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
        )
        self._world_binding.initialize()
        self._world_binding.get_world_interface().update_world_to_robot_root_transforms(poses=(robot_pos, robot_ori))
        self._world_binding.synchronize_transforms()

    def _build_controller(self) -> None:
        """Create the cuMotion RMPflow controller for the UR10 arm."""
        if self._robot is None or self._world_binding is None:
            return

        self._cumotion_robot = load_cumotion_supported_robot("ur10")
        self._site_space = self._cumotion_robot.robot_description.tool_frame_names()
        self._tool_frame = "tool0"
        if self._tool_frame not in self._site_space:
            raise RuntimeError("cuMotion UR10 configuration does not provide 'tool0'.")
        self._controller = RmpFlowController(
            cumotion_robot=self._cumotion_robot,
            cumotion_world_interface=self._world_binding.get_world_interface(),
            robot_joint_space=self._robot.dof_names,
            robot_site_space=self._site_space,
            tool_frame=self._tool_frame,
        )

    def _get_estimated_state(self) -> mg.RobotState:
        """Wrap the robot's current joint state for the motion-generation controller.

        Returns:
            Current robot joint state in the controller's joint-space format.
        """
        names = self._robot.dof_names
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=names,
                positions=(names, self._robot.get_dof_positions()),
                velocities=(names, self._robot.get_dof_velocities()),
            )
        )

    def _create_setpoint_state(self, target_position: np.ndarray) -> mg.RobotState:
        """Build the desired tool-frame state for RMPflow.

        Args:
            target_position: Desired tool-frame position in world coordinates.

        Returns:
            Desired tool-frame state for the controller.
        """
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=self._site_space,
                positions=([self._tool_frame], wp.array([target_position.tolist()], dtype=wp.float32)),
                orientations=([self._tool_frame], wp.array([self._tool_orientation.tolist()], dtype=wp.float32)),
            ),
        )

    def _move_to_target(self, target_position: np.ndarray, dt: float) -> None:
        """Advance RMPflow toward the requested tool-frame target.

        Args:
            target_position: Desired tool-frame position in world coordinates.
            dt: Physics step size in seconds.
        """
        if (
            self._robot is None
            or self._controller is None
            or self._world_binding is None
            or self._site_space is None
            or self._tool_frame is None
            or self._tool_orientation is None
        ):
            return

        estimated = self._get_estimated_state()
        setpoint = self._create_setpoint_state(target_position)
        if self._rmpflow_reset_needed:
            self._motion_generation_t = 0.0
            if not self._controller.reset(estimated, setpoint, t=self._motion_generation_t):
                raise RuntimeError("RmpFlowController reset failed.")
            self._rmpflow_reset_needed = False
        else:
            self._motion_generation_t += dt

        self._world_binding.get_world_interface().update_world_to_robot_root_transforms(self._robot.get_world_poses())
        self._world_binding.synchronize_transforms()
        desired = self._controller.forward(estimated, setpoint, self._motion_generation_t)
        if desired is None or desired.joints is None:
            return

        if desired.joints.positions is not None:
            self._robot.set_dof_position_targets(
                positions=desired.joints.positions,
                dof_indices=desired.joints.position_indices,
            )
        if desired.joints.velocities is not None:
            self._robot.set_dof_velocity_targets(
                velocities=desired.joints.velocities,
                dof_indices=desired.joints.velocity_indices,
            )

    # ------------------------------------------------------------------
    # Cube pool management
    # ------------------------------------------------------------------

    def _create_cube_pool(self) -> None:
        """Pre-create cubes off-screen, hidden, with collision and rigid body APIs.

        Cubes are kept physics-enabled (to avoid tensor view errors on disabled
        bodies) but positioned far off-screen and invisible. Velocities are
        zeroed when a cube is activated.
        """
        offscreen = [0.0, 0.0, -1000.0]
        for i in range(MAX_CUBES):
            path = f"/World/cube_{i}"
            Cube(path, sizes=CUBE_SIZE, positions=offscreen, colors=[0.6, 0.6, 0.6])
            GeomPrim(path, apply_collision_apis=True)

        cube_paths = [f"/World/cube_{i}" for i in range(MAX_CUBES)]
        self._cube_pool = RigidPrim(cube_paths, resolve_paths=False)
        self._cube_pool.set_visibilities(False)
        self._active_cubes = 0

    def _deactivate_all_cubes(self) -> None:
        """Hide all active cubes and move them off-screen."""
        if self._cube_pool is None or self._active_cubes == 0:
            return
        indices = list(range(self._active_cubes))
        self._cube_pool.set_visibilities(False, indices=indices)
        self._cube_pool.set_world_poses(positions=[0.0, 0.0, -1000.0], indices=indices)
        self._active_cubes = 0
        self._cubes_to_add = 0

    def _activate_next_cube(self) -> None:
        """Activate one cube from the pool at the pipe position with a random orientation."""
        if self._cube_pool is None or self._active_cubes >= len(self._cube_pool):
            self._cubes_to_add = 0
            return

        orientation = np.array([random.random(), random.random(), random.random(), random.random()])
        orientation = orientation / np.linalg.norm(orientation)

        idx = self._active_cubes
        self._cube_pool.set_world_poses(positions=PIPE_POSITION, orientations=orientation, indices=idx)
        self._cube_pool.set_visibilities(True, indices=idx)
        self._cube_pool.set_velocities(
            linear_velocities=[0.0, 0.0, 0.0], angular_velocities=[0.0, 0.0, 0.0], indices=idx
        )

        self._active_cubes += 1
        self._cubes_to_add -= 1

    # ------------------------------------------------------------------
    # Start event (called from extension UI)
    # ------------------------------------------------------------------

    async def on_fill_bin_event_async(self) -> None:
        """Start the bin filling simulation by registering the physics callback."""
        if self._robot is None:
            return

        if not app_utils.is_playing():
            app_utils.play()
            await app_utils.update_app_async()

        self._rmpflow_reset_needed = True
        self._physics_callback_id = SimulationManager.register_callback(
            self._physics_step, event=SimulationEvent.PHYSICS_POST_STEP
        )

    # ------------------------------------------------------------------
    # 7-phase pick-and-hold state machine
    # ------------------------------------------------------------------

    def _physics_step(self, dt: float, context: object) -> None:
        """Execute one step of the pick-and-hold state machine and handle cube spawning.

        Args:
            dt: Physics step size in seconds.
            context: Callback context supplied by the simulation manager.
        """
        if self._robot is None or self._bin_prim is None:
            return

        if self._is_done():
            return

        if (
            self._cubes_to_add > 0
            and self._cube_pool is not None
            and self._active_cubes < len(self._cube_pool)
            and self._step_index % 30 == 0
        ):
            self._activate_next_cube()
        self._step_index += 1

        if self._pause:
            return

        bin_pos, _ = self._bin_prim.get_world_poses()
        bin_position = bin_pos.numpy().flatten()
        placing_position = np.asarray(TARGET_POSITION)

        if self._event in [0, 1, 2, 3]:
            self._pick_position = bin_position.copy()
            self._pick_height = self._pick_position[2]
        picking_position = self._pick_position if self._pick_position is not None else bin_position

        if self._event == 2:
            pass
        elif self._event == 3:
            self._surface_gripper_iface.close_gripper(self._surface_gripper_path)
        else:
            if self._event < 5:
                target_xy = picking_position[:2]
            elif self._event == 5:
                target_xy = self._combine_convex(picking_position[:2], placing_position[:2], self._mix_sin(self._t))
            else:
                target_xy = placing_position[:2]

            if self._event == 1:
                target_height = self._combine_convex(INITIAL_EE_HEIGHT, self._pick_height, self._mix_sin(self._t))
            elif self._event == 4:
                target_height = self._combine_convex(self._pick_height, INITIAL_EE_HEIGHT, self._mix_sin(self._t))
            else:
                target_height = INITIAL_EE_HEIGHT

            suction_target = np.array(
                [
                    target_xy[0] + EE_OFFSET[0],
                    target_xy[1] + EE_OFFSET[1],
                    target_height + EE_OFFSET[2],
                ]
            )
            ee_link_target = suction_target - self._suction_offset_world

            self._move_to_target(ee_link_target, dt)

        self._t += EVENTS_DT[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0.0
            self._rmpflow_reset_needed = True

        if not self._added_cubes and self._event == 6 and not self._pause:
            self._pause = True
            self._cubes_to_add += 20
            self._added_cubes = True

        if self._is_done():
            app_utils.pause()

    def _is_done(self) -> bool:
        """Return whether the state machine has completed all phases.

        Returns:
            True if all state-machine phases have completed.
        """
        return self._event >= len(EVENTS_DT)

    @staticmethod
    def _mix_sin(t: float) -> float:
        """Return a smooth 0-1 interpolation value.

        Args:
            t: Normalized interpolation phase.

        Returns:
            Smooth interpolation value between 0 and 1.
        """
        return 0.5 * (1.0 - np.cos(t * np.pi))

    @staticmethod
    def _combine_convex(a: np.ndarray | float, b: np.ndarray | float, alpha: float) -> np.ndarray | float:
        """Blend two values with a convex combination.

        Args:
            a: Start value.
            b: End value.
            alpha: Blend weight applied to ``b``.

        Returns:
            Convex blend of ``a`` and ``b``.
        """
        return (1.0 - alpha) * a + alpha * b

    @staticmethod
    def _get_tool_orientation_from_ee_orientation(ee_orientation: np.ndarray) -> np.ndarray:
        """Convert the desired USD ee-link orientation to the cuMotion tool orientation.

        Args:
            ee_orientation: Desired ee-link orientation.

        Returns:
            Normalized cuMotion tool orientation.
        """
        ee_link_from_wrist = euler_angles_to_quaternion([0.0, 0.0, np.pi / 2.0]).numpy()
        tool_from_wrist = euler_angles_to_quaternion([-np.pi / 2.0, 0.0, 0.0]).numpy()
        wrist_from_ee_link = quaternion_conjugate(ee_link_from_wrist).numpy()
        tool_orientation = quaternion_multiplication(
            quaternion_multiplication(ee_orientation, wrist_from_ee_link), tool_from_wrist
        ).numpy()
        return tool_orientation / np.linalg.norm(tool_orientation)
