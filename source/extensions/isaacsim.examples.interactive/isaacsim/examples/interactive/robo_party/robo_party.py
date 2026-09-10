"""Robo party sample combining Franka stacking, UR10 stacking, Kaya, and Jetbot."""

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

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.controllers as motion_controllers
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.examples.base import BaseSample
from isaacsim.robot.experimental.wheeled_robots.robots import HolonomicRobotUsdSetup
from isaacsim.robot_motion.examples.manipulation import PickPlaceTask
from isaacsim.storage.native import get_assets_root_path


class RoboParty(BaseSample):
    """Interactive sample: Franka stacking + UR10 stacking + Kaya (holonomic) + Jetbot (differential)."""

    def __init__(self) -> None:
        super().__init__()
        self._stacking: PickPlaceTask | None = None
        self._ur10_stacking: PickPlaceTask | None = None
        self._kaya: Articulation | None = None
        self._jetbot: Articulation | None = None
        self._holonomic_controller: motion_controllers.HolonomicController | None = None
        self._differential_controller: motion_controllers.DifferentialDriveController | None = None
        self._physics_callback_id: int | None = None
        self._is_executing = False
        self._party_step = 0

    def setup_scene(self) -> None:
        """Set up the scene: Franka stacking, UR10 stacking, Kaya, and Jetbot."""
        # Franka stacking
        self._stacking = PickPlaceTask(
            robot_path="/World/robot_0",
            cube_path="/World/FrankaCube",
            cube_positions=[
                (0.3, 0.3, 0.025),
                (0.3, -0.3, 0.025),
            ],
            offset=(0.0, -2.0, 0.0),
            robot_name="franka",
        )
        self._stacking.setup_scene()

        # UR10 stacking
        self._ur10_stacking = PickPlaceTask(
            robot_path="/World/robot_1",
            cube_path="/World/Ur10Cube",
            cube_positions=[
                (0.45, -0.2, 0.025),
                (0.45, -0.4, 0.025),
            ],
            place_position=(0.45, 0.05, 0.0258),
            offset=(0.0, 0.0, 0.0),
            robot_name="ur10",
        )
        self._ur10_stacking.setup_scene()

        assets_root = get_assets_root_path()
        if assets_root is None:
            return

        # Kaya (holonomic)
        kaya_path = "/World/Kaya"
        kaya_usd = assets_root + "/Isaac/Robots_Multiphysics/NVIDIA/Kaya/kaya.usda"
        stage_utils.add_reference_to_stage(usd_path=kaya_usd, path=kaya_path)
        self._kaya = Articulation(
            kaya_path,
            positions=np.array([-1.0, 1.0, 0.02]),
            reset_xform_op_properties=True,
        )

        # Jetbot (differential)
        jetbot_path = "/World/Jetbot"
        jetbot_usd = assets_root + "/Isaac/Robots_Multiphysics/NVIDIA/Jetbot/jetbot.usda"
        stage_utils.add_reference_to_stage(usd_path=jetbot_usd, path=jetbot_path)
        self._jetbot = Articulation(
            jetbot_path,
            positions=np.array([-1.5, -1.5, 0.0]),
            reset_xform_op_properties=True,
        )

    async def setup_post_load(self) -> None:
        """Build controllers for Kaya and Jetbot after scene is loaded."""
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()
        await app_utils.update_app_async()

        if self._stacking is not None and self._ur10_stacking is not None:
            self._stacking.initialize(
                ("/World/robot_1", "/World/Kaya", "/World/Jetbot", *self._ur10_stacking.cube_paths)
            )
            self._ur10_stacking.initialize(
                ("/World/robot_0", "/World/Kaya", "/World/Jetbot", *self._stacking.cube_paths)
            )

        # View so Franka/UR10 stackings and Jetbot/Kaya are all visible (target slightly toward wheeled robots)
        ViewportManager.set_camera_view(eye=[10.0, 0.0, 5.0], target=[0.0, -2.0, 0.0], camera="/OmniverseKit_Persp")

        if self._kaya is not None:
            kaya_setup = HolonomicRobotUsdSetup(
                robot_prim_path="/World/Kaya",
                com_prim_path="/World/Kaya/base_link/control_offset",
            )
            (
                wheel_radius,
                wheel_positions,
                wheel_orientations,
                mecanum_angles,
                wheel_axis,
                up_axis,
            ) = kaya_setup.get_holonomic_controller_params()
            self._holonomic_controller = motion_controllers.HolonomicController(
                robot_joint_space=list(self._kaya.dof_names),
                wheel_joint_names=kaya_setup.get_articulation_controller_params(),
                wheel_radius=wheel_radius,
                wheel_positions=wheel_positions,
                wheel_orientations=wheel_orientations,
                mecanum_angles=mecanum_angles,
                wheel_axis=wheel_axis,
                rotation_direction=up_axis,
                device=SimulationManager.get_physics_sim_device(),
            )

        if self._jetbot is not None:
            self._differential_controller = motion_controllers.DifferentialDriveController(
                robot_joint_space=list(self._jetbot.dof_names),
                left_wheel_joint="left_wheel_joint",
                right_wheel_joint="right_wheel_joint",
                wheel_radius=0.03,
                wheel_base=0.1125,
                device=SimulationManager.get_physics_sim_device(),
            )

    async def setup_pre_reset(self) -> None:
        """Remove physics callback and reset state."""
        self._remove_physics_callback()
        if self._stacking is not None:
            self._stacking.reset()
        if self._ur10_stacking is not None:
            self._ur10_stacking.reset()
        self._is_executing = False
        self._party_step = 0

    async def setup_post_reset(self) -> None:
        """After reset: reset Franka and UR10 to default pose."""
        if self._stacking is not None:
            self._stacking.reset_robot()
        if self._ur10_stacking is not None:
            self._ur10_stacking.reset_robot()

    async def setup_post_clear(self) -> None:
        """Clear all references."""
        self._remove_physics_callback()
        if self._stacking is not None:
            self._stacking.cleanup()
        if self._ur10_stacking is not None:
            self._ur10_stacking.cleanup()
        self._stacking = None
        self._ur10_stacking = None
        self._kaya = None
        self._jetbot = None
        self._holonomic_controller = None
        self._differential_controller = None
        self._is_executing = False
        self._party_step = 0

    def physics_cleanup(self) -> None:
        """Clean up physics callback and state."""
        self._remove_physics_callback()
        if self._stacking is not None:
            self._stacking.cleanup()
        if self._ur10_stacking is not None:
            self._ur10_stacking.cleanup()
        self._stacking = None
        self._ur10_stacking = None
        self._kaya = None
        self._jetbot = None
        self._holonomic_controller = None
        self._differential_controller = None
        self._is_executing = False
        self._party_step = 0

    def _remove_physics_callback(self) -> None:
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

    def _party_physics_callback(self, dt: float, context: object) -> None:
        """Run stacking and time-based wheeled robot commands.

        Args:
            dt: Time delta for the physics step.
            context: Physics simulation context.
        """
        if not self._is_executing:
            return

        # Franka stacking
        if self._stacking is not None and not self._stacking.is_done and not self._stacking.failed:
            self._stacking.step(dt)

        # UR10 stacking
        if self._ur10_stacking is not None and not self._ur10_stacking.is_done and not self._ur10_stacking.failed:
            self._ur10_stacking.step(dt)

        # Time-based commands for Kaya and Jetbot (same idea as original robo_party)
        if self._party_step < 500:
            kaya_cmd = np.array([0.2, 0.0, 0.0])
            jetbot_cmd = np.array([0.1, 0.0])
        elif self._party_step < 600:
            kaya_cmd = np.array([-0.2, 0.0, 0.0])
            jetbot_cmd = np.array([0.0, np.pi / 10])
        elif self._party_step < 900:
            kaya_cmd = np.array([0.0, 0.0, 0.06])
            jetbot_cmd = np.array([0.1, 0.0])
        else:
            kaya_cmd = np.array([0.0, 0.0, 0.0])
            jetbot_cmd = np.array([0.0, 0.0])

        if self._holonomic_controller is not None and self._kaya is not None:
            self._apply_drive_command(
                self._kaya, self._holonomic_controller, kaya_cmd[:2], angular_velocity=kaya_cmd[2], dt=dt
            )
        if self._differential_controller is not None and self._jetbot is not None:
            self._apply_drive_command(
                self._jetbot,
                self._differential_controller,
                (jetbot_cmd[0], 0.0),
                angular_velocity=jetbot_cmd[1],
                dt=dt,
            )

        self._party_step += 1

    def _apply_drive_command(
        self,
        robot: Articulation,
        controller: motion_controllers.HolonomicController | motion_controllers.DifferentialDriveController,
        linear_velocity: tuple[float, float] | np.ndarray,
        angular_velocity: float,
        dt: float,
    ) -> None:
        """Apply a planar velocity command through a robot-motion controller.

        Args:
            robot: Articulation to drive.
            controller: Holonomic or differential-drive controller for the robot.
            linear_velocity: Desired planar linear velocity (x, y) in m/s.
            angular_velocity: Desired yaw rate in rad/s.
            dt: Simulation step size in seconds.
        """
        site_name = "control_point"
        device = SimulationManager.get_physics_sim_device()
        setpoint = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=[site_name],
                linear_velocities=(
                    [site_name],
                    wp.array([[linear_velocity[0], linear_velocity[1], 0.0]], dtype=wp.float32, device=device),
                ),
                angular_velocities=(
                    [site_name],
                    wp.array([[0.0, 0.0, angular_velocity]], dtype=wp.float32, device=device),
                ),
            )
        )
        desired = controller.forward(mg.RobotState(), setpoint, self._party_step * dt)
        if desired is not None and desired.joints is not None:
            robot.set_dof_velocity_targets(
                desired.joints.velocities,
                dof_indices=robot.get_dof_indices(desired.joints.velocity_names),
            )

    async def _on_start_party_event_async(self) -> None:
        """Start the party: register physics callback and play."""
        if self._is_executing:
            return
        if self._stacking is None and self._ur10_stacking is None:
            return
        if self._stacking is not None:
            self._stacking.reset()
        if self._ur10_stacking is not None:
            self._ur10_stacking.reset()
        self._is_executing = True
        self._party_step = 0
        self._physics_callback_id = SimulationManager.register_callback(
            self._party_physics_callback, event=SimulationEvent.PHYSICS_POST_STEP
        )
        app_utils.play()
        await app_utils.update_app_async()
