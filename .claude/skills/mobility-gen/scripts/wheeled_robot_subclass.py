"""Example WheeledMobilityGenRobot subclass for custom differential-drive robots.

Demonstrates the class-attribute-only pattern for registering a custom wheeled
robot with MobilityGen without overriding build() or write_action().
"""


def make_wheeled_robot_class():
    """Return MyRobot class with MobilityGen dependencies resolved.

    Call this at runtime when Isaac Sim is available, then pass the returned
    class to MobilityGen's recording/replay pipeline.
    """
    from isaacsim.replicator.experimental.mobility_gen import ROBOTS
    from isaacsim.replicator.mobility_gen.examples.misc import HawkCamera
    from isaacsim.replicator.mobility_gen.examples.robots import WheeledMobilityGenRobot

    @ROBOTS.register()
    class MyRobot(WheeledMobilityGenRobot):
        """Custom differential-drive robot for MobilityGen SDG.

        Set all configuration as class attributes; no method overrides needed
        for standard wheeled differential-drive robots.
        """

        physics_dt: float = 0.005
        z_offset: float = 0.25

        chase_camera_base_path = "chassis"
        chase_camera_x_offset: float = -1.5
        chase_camera_z_offset: float = 0.8
        chase_camera_tilt_angle: float = 60.0

        front_camera_base_path = "chassis/front_hawk"
        front_camera_rotation = (0.0, 0.0, 0.0)
        front_camera_translation = (0.2, 0.0, 0.1)
        front_camera_type = HawkCamera

        occupancy_map_radius: float = 0.5
        occupancy_map_z_min: float = 0.1
        occupancy_map_z_max: float = 0.5
        occupancy_map_cell_size: float = 0.05
        occupancy_map_collision_radius: float = 0.5

        keyboard_linear_velocity_gain: float = 1.0
        keyboard_angular_velocity_gain: float = 1.0
        gamepad_linear_velocity_gain: float = 1.0
        gamepad_angular_velocity_gain: float = 1.0

        random_action_linear_velocity_range = (-0.3, 1.0)
        random_action_angular_velocity_range = (-0.75, 0.75)
        random_action_linear_acceleration_std: float = 5.0
        random_action_angular_acceleration_std: float = 5.0
        random_action_grid_pose_sampler_grid_size: float = 5.0
        path_following_speed: float = 1.0
        path_following_angular_gain: float = 1.0
        path_following_stop_distance_threshold: float = 0.5
        path_following_forward_angle_threshold = 0.785
        path_following_target_point_offset_meters: float = 1.0

        wheel_dof_names = ["left_wheel_joint", "right_wheel_joint"]
        usd_url: str = "/path/to/my_robot.usd"
        chassis_subpath: str = "chassis"
        wheel_base: float = 0.5
        wheel_radius: float = 0.1

    return MyRobot
