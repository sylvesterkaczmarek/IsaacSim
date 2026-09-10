"""Example holonomic (3-wheel) MobilityGenRobot subclass (Kaya).

Demonstrates overriding build() and write_action() for a holonomic drive robot
that uses HolonomicController instead of a differential drive controller.
"""


def make_kaya_robot_class():
    """Return KayaRobot class with MobilityGen + holonomic dependencies resolved.

    Call this at runtime when Isaac Sim is available, then pass the returned
    class to MobilityGen's recording/replay pipeline.
    """
    import numpy as np
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.experimental.utils.prim import join_prim_paths
    from isaacsim.core.experimental.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.replicator.experimental.mobility_gen import ROBOTS
    from isaacsim.replicator.mobility_gen.examples.misc import HawkCamera
    from isaacsim.replicator.mobility_gen.examples.robots import WheeledMobilityGenRobot
    from isaacsim.robot.experimental.wheeled_robots.controllers.holonomic_controller import HolonomicController
    from isaacsim.robot.experimental.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
    from isaacsim.storage.native import get_assets_root_path

    @ROBOTS.register()
    class KayaRobot(WheeledMobilityGenRobot):
        """Holonomic 3-wheel robot (NVIDIA Kaya) for MobilityGen SDG.

        Overrides build() to configure a HolonomicController and write_action()
        to map the 2D [lin, ang] action to the [forward, lateral, yaw] command.
        """

        physics_dt: float = 0.005
        z_offset: float = 0.02
        chase_camera_base_path = "base_link"
        chase_camera_x_offset: float = -0.5
        chase_camera_z_offset: float = 0.3
        chase_camera_tilt_angle: float = 60.0
        front_camera_base_path = "base_link/front_hawk"
        front_camera_rotation = (0.0, 0.0, 0.0)
        front_camera_translation = (0.1, 0.0, 0.05)
        front_camera_type = HawkCamera

        occupancy_map_radius: float = 0.2
        occupancy_map_z_min: float = 0.02
        occupancy_map_z_max: float = 0.3
        occupancy_map_cell_size: float = 0.05
        occupancy_map_collision_radius: float = 0.2
        random_action_linear_velocity_range = (-0.2, 0.4)
        random_action_angular_velocity_range = (-0.5, 0.5)
        random_action_linear_acceleration_std: float = 1.0
        random_action_angular_acceleration_std: float = 2.0
        random_action_grid_pose_sampler_grid_size: float = 5.0
        path_following_speed: float = 0.4
        path_following_angular_gain: float = 1.0
        path_following_stop_distance_threshold: float = 0.3
        path_following_forward_angle_threshold = 0.785
        path_following_target_point_offset_meters: float = 0.5
        keyboard_linear_velocity_gain: float = 0.4
        keyboard_angular_velocity_gain: float = 0.5
        gamepad_linear_velocity_gain: float = 0.4
        gamepad_angular_velocity_gain: float = 0.5

        wheel_dof_names = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
        usd_url: str = None  # set at runtime: get_assets_root_path() + "/Isaac/Robots/NVIDIA/Kaya/kaya.usd"
        chassis_subpath: str = "base_link"
        wheel_radius: float = 0.04
        wheel_base: float = 0.1
        com_prim_subpath: str = "base_link/control_offset"

        def __init__(
            self,
            prim_path: str,
            articulation: Articulation,
            controller: HolonomicController,
            front_camera=None,
        ) -> None:
            # WheeledMobilityGenRobot annotates `controller` as a
            # DifferentialController.  Re-annotate it here so type checkers see
            # the holonomic controller this robot actually drives.
            super().__init__(
                prim_path=prim_path, articulation=articulation, controller=controller, front_camera=front_camera
            )

        @classmethod
        def build(cls, prim_path: str):
            if cls.usd_url is None:
                cls.usd_url = get_assets_root_path() + "/Isaac/Robots/NVIDIA/Kaya/kaya.usd"
            add_reference_to_stage(usd_path=cls.usd_url, path=prim_path)
            get_current_stage(backend="usd").Load(prim_path)
            articulation = Articulation(prim_path)
            kaya_setup = HolonomicRobotUsdSetup(
                robot_prim_path=prim_path,
                com_prim_path=join_prim_paths(prim_path, cls.com_prim_subpath),
            )
            wheel_radius, wheel_positions, wheel_orientations, mecanum_angles, wheel_axis, up_axis = (
                kaya_setup.get_holonomic_controller_params()
            )
            controller = HolonomicController(
                name="kaya_controller",
                wheel_radius=wheel_radius,
                wheel_positions=wheel_positions,
                wheel_orientations=wheel_orientations,
                mecanum_angles=mecanum_angles,
                wheel_axis=wheel_axis,
                up_axis=up_axis,
            )
            camera = cls.build_front_camera(prim_path)
            return cls(prim_path=prim_path, articulation=articulation, controller=controller, front_camera=camera)

        def write_action(self, step_size: float) -> None:
            # Mirrors WheeledMobilityGenRobot.write_action, but remaps the 2D
            # [linear, angular] action to the holonomic [forward, lateral, yaw].
            if not self.is_physics_ready():
                return
            if self._wheel_indices is None:
                self._wheel_indices = self.articulation.get_joint_indices(self.wheel_dof_names)
            action = self.action.get_value()
            velocities = self.controller.forward(command=[action[0], 0.0, action[1]])
            self.articulation.set_dof_velocities(velocities[np.newaxis], dof_indices=self._wheel_indices)

    return KayaRobot
